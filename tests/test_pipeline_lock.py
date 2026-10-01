"""Tests for PipelineLock — flock-based process-level lock.

核心承诺：进程死了内核放锁。这无法在单进程内模拟（同一进程的 flock
探测 fd 与持锁 fd 互斥，但进程退出释放锁是内核行为），所以用
multiprocessing 子进程 + SIGKILL 直接验证。
"""

import os
import signal
import sys
import time
from multiprocessing import Process
from pathlib import Path

import pytest

from utils.pipeline_lock import PipelineLock

HAS_FCNTL = sys.platform != "win32"
skip_no_fcntl = pytest.mark.skipif(not HAS_FCNTL, reason="fcntl only on Unix")


@skip_no_fcntl
def test_acquire_then_is_held(tmp_path):
    lock = PipelineLock(tmp_path / "pipeline.lock")
    assert lock.is_held() is False
    assert lock.acquire() is True
    assert lock.is_held() is True


@skip_no_fcntl
def test_release_clears_lock(tmp_path):
    lock = PipelineLock(tmp_path / "pipeline.lock")
    lock.acquire()
    assert lock.is_held() is True
    lock.release()
    assert lock.is_held() is False


@skip_no_fcntl
def test_second_acquire_fails_while_held(tmp_path):
    """同一文件，第二个实例持不到第一把锁。"""
    lock1 = PipelineLock(tmp_path / "pipeline.lock")
    lock2 = PipelineLock(tmp_path / "pipeline.lock")
    assert lock1.acquire() is True
    assert lock2.acquire() is False  # 被持，非阻塞返回 False
    lock1.release()


@skip_no_fcntl
def test_acquire_is_idempotent(tmp_path):
    """同一实例重复 acquire 不开新 fd、不报错。"""
    lock = PipelineLock(tmp_path / "pipeline.lock")
    assert lock.acquire() is True
    assert lock.acquire() is True  # 幂等短路
    lock.release()
    assert lock.is_held() is False


@skip_no_fcntl
def test_release_is_idempotent(tmp_path):
    """未持锁时 release 不抛错。"""
    lock = PipelineLock(tmp_path / "pipeline.lock")
    lock.release()  # 未持，不抛
    lock.release()  # 再调一次也不抛


@skip_no_fcntl
def test_is_held_does_not_steal_lock(tmp_path):
    """is_held() 是纯探测：调完不改变持锁状态。"""
    lock = PipelineLock(tmp_path / "pipeline.lock")
    lock.acquire()
    # 另一个实例探测
    probe = PipelineLock(tmp_path / "pipeline.lock")
    assert probe.is_held() is True
    # 探测完，lock1 仍持锁（probe 没偷走）
    assert lock.is_held() is True
    # 第二个实例仍抢不到
    assert probe.acquire() is False
    lock.release()


# --- 子进程辅助：在子进程里持锁，父进程观察 ---

def _child_hold(lock_path, ready_event_path):
    """子进程：抢锁后写 ready 文件通知父进程，然后阻塞不退出。"""
    lock = PipelineLock(lock_path)
    if not lock.acquire():
        # 抢不到说明父进程误判，写错误标记
        Path(ready_event_path).write_text("FAIL")
        return
    Path(ready_event_path).write_text("READY")
    # 阻塞，保持持锁，直到被父进程 SIGKILL
    while True:
        time.sleep(0.1)


@skip_no_fcntl
def test_process_death_releases_lock(tmp_path):
    """核心承诺：子进程被 SIGKILL 后，内核自动释放锁。

    这是整个 flock 方案的价值所在——进程死了不靠善后代码，内核放锁。
    """
    lock_path = tmp_path / "pipeline.lock"
    ready_path = tmp_path / "ready"

    child = Process(target=_child_hold, args=(lock_path, str(ready_path)))
    child.start()

    # 等子进程拿到锁并写 ready（macOS 默认 spawn 启动较慢，给足 15s）
    deadline = time.time() + 15
    while time.time() < deadline:
        if ready_path.exists():
            break
        if not child.is_alive():
            pytest.fail(f"子进程提前退出，exitcode={child.exitcode}")
        time.sleep(0.1)
    assert ready_path.exists(), "子进程未在超时内就绪"
    assert ready_path.read_text() == "READY", "子进程抢锁失败"

    # 父进程观察：锁被持
    parent_lock = PipelineLock(lock_path)
    assert parent_lock.is_held() is True

    # 杀子进程（SIGKILL，无善后机会）
    os.kill(child.pid, signal.SIGKILL)
    child.join(timeout=5)
    assert not child.is_alive(), "子进程未被杀掉"

    # 内核应已释放锁——父进程现在能探测到"没人持锁"
    assert parent_lock.is_held() is False
    # 而且能成功抢到
    assert parent_lock.acquire() is True
    parent_lock.release()
