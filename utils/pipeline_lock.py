"""Pipeline 进程级排他锁：用 flock 把"后台搜索是否在跑"的判定外包给内核。

WHY
当前 pipeline 的"是否在跑"靠 ``pipeline_progress.json`` 的 ``done`` 字段
（app.py 线程 finally 里写 ``done=True``）。一旦 Streamlit 进程被强杀
（Ctrl+C / kill -9 / 断电 / daemon 线程被宿主进程丢弃），finally 跑不到，
文件就停在 ``done:false``，下次启动按钮误判"搜索中"——这就是"僵尸进度"。

flock 是内核维护、绑定到进程生命周期的排他锁：进程退出（含 SIGKILL、
断电）时内核自动释放，**不依赖任何善后代码**。把死活判定从"程序汇报"
换成"内核记账"，根治僵尸进度。

PLATFORM
``fcntl.flock`` 在 macOS / Linux 本地磁盘上行为正确（项目部署目标是
Unix 服务器）。NFS 等网络文件系统有坑，但本工具的 runtime 目录始终
在本地磁盘，不受影响。

PROCESS-LOCAL SINGLETON
PipelineLock 实例由 ``app.py`` 的 ``@st.cache_resource`` 缓存为进程级单例，
所以"实例字段 ``self._fd`` 当幂等状态记忆"在 Streamlit rerun 下成立：
rerun 不会 new 新实例，``_fd`` 跨 rerun 保持。少了单例这步，幂等判断
会失效（每次 rerun 新实例的 ``_fd`` 都是 None，重复 acquire 会开多个 fd）。
"""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path


class PipelineLock:
    """进程级排他锁，靠 flock 实现。

    持锁 fd 由本进程保持打开；进程退出（含 SIGKILL/断电）内核自动释放。
    ``is_held()`` 用一个独立的探测 fd 去尝试加锁：拿得到→没人持锁；
    拿不到→有人持锁。flock 锁绑定到 open file description，所以同一进程
    的探测 fd 与持锁 fd 也会冲突，``is_held()`` 在"本进程自己持锁"时
    也正确返回 True。
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._fd: int | None = None  # 持锁 fd；None 表示未持锁

    def acquire(self) -> bool:
        """非阻塞抢锁。成功返回 True；已被持返回 False。

        幂等：对同一实例重复调用不会开新 fd（避免同一进程对同一文件
        持多份互不冲突的锁，导致 release 漏放）。
        """
        if self._fd is not None:
            return True  # 已持锁，幂等短路
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as e:
            os.close(fd)
            # EWOULDBLOCK / EAGAIN = 别人持着，是预期分支，不报错
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                return False
            raise  # 其他 OSError（权限/磁盘满）向上抛，不静默吞掉
        self._fd = fd
        return True

    def release(self) -> None:
        """释放锁。幂等：未持锁时调用不抛错。"""
        if self._fd is None:
            return  # 幂等
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def is_held(self) -> bool:
        """内核级死活探测：是否有任何进程正持锁。不改变持锁状态。

        实现用一个临时探测 fd 去尝试非阻塞加锁——拿得到说明没人持，
        立即放回；拿不到说明有人持（可能就是本进程自己）。
        """
        if self._fd is not None:
            return True  # 自己持着，直接返回
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as e:
                if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    return True  # 别的进程持着
                raise
            # 拿到了说明没人持，立即放回，保持无锁状态
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)
