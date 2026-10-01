"""Tests for CircuitBreaker."""

import time

import pytest

from utils.llm_circuit_breaker import CircuitBreaker


def test_no_cooldown_by_default():
    b = CircuitBreaker()
    assert b.is_cooled_down("p1") is False
    assert b.remaining("p1") == 0.0


def test_failure_marks_cooldown():
    b = CircuitBreaker(cooldown_seconds=60)
    b.record_failure("p1")
    assert b.is_cooled_down("p1") is True
    assert b.remaining("p1") > 0


def test_success_clears_cooldown():
    b = CircuitBreaker(cooldown_seconds=60)
    b.record_failure("p1")
    assert b.is_cooled_down("p1") is True
    b.record_success("p1")
    assert b.is_cooled_down("p1") is False
    assert b.remaining("p1") == 0.0


def test_cooldown_expires(monkeypatch):
    """Cooldown clears after cooldown_seconds elapse."""
    b = CircuitBreaker(cooldown_seconds=60)
    t = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])

    b.record_failure("p1")
    t[0] = 1059.0  # 59s later — still cooled down
    assert b.is_cooled_down("p1") is True
    assert b.remaining("p1") == pytest.approx(1.0)

    t[0] = 1061.0  # 61s later — expired
    assert b.is_cooled_down("p1") is False
    assert b.remaining("p1") == 0.0


def test_remaining_decreases_over_time(monkeypatch):
    b = CircuitBreaker(cooldown_seconds=60)
    t = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])

    b.record_failure("p1")
    assert b.remaining("p1") == pytest.approx(60.0)
    t[0] = 1030.0
    assert b.remaining("p1") == pytest.approx(30.0)


def test_snapshot(monkeypatch):
    b = CircuitBreaker(cooldown_seconds=60)
    t = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    b.record_failure("p1")
    b.record_failure("p2")
    snap = b.snapshot()
    assert set(snap.keys()) == {"p1", "p2"}
    assert all(v > 0 for v in snap.values())


def test_snapshot_excludes_expired(monkeypatch):
    b = CircuitBreaker(cooldown_seconds=60)
    t = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    b.record_failure("p1")
    t[0] = 1061.0  # p1 expired
    snap = b.snapshot()
    assert snap == {}


def test_independent_presets():
    b = CircuitBreaker()
    b.record_failure("p1")
    assert b.is_cooled_down("p1") is True
    assert b.is_cooled_down("p2") is False
    b.record_success("p1")
    b.record_failure("p2")
    assert b.is_cooled_down("p1") is False
    assert b.is_cooled_down("p2") is True
