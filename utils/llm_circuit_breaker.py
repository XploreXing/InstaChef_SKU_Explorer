"""Circuit breaker: tracks failed presets and cools them down for a window.

WHY
When a preset (provider+model) fails with a transient error (timeout / 429 /
5xx / auth), retrying it immediately on the next call wastes latency and
hammers an already-down endpoint. The breaker marks it "cooled down" for
`cooldown_seconds`; the router's RoutingStrategy skips cooled-down presets
when selecting the attempt order. After the window expires, the preset is
retried. A success clears the cooldown early.

STREAMLIT PERSISTENCE
Streamlit re-runs the whole script on every interaction. A breaker
instantiated locally inside a function would lose its `_cooldowns` state on
every rerun, defeating the purpose. The breaker must be a process-wide
singleton — in app.py it's cached via `@st.cache_resource` and injected into
every LLMRouter, so cooldown state survives across reruns and sessions.
(Outside Streamlit, e.g. unit tests, the router self-constructs a breaker.)

TIME BASIS
Uses time.monotonic() (not wall-clock time.time()) so cooldowns aren't
affected by system clock jumps (NTP syncs, DST, manual changes).
"""

from __future__ import annotations

import time


class CircuitBreaker:
    """Per-preset cooldown tracker. Thread-safe via a lock around dict ops."""

    def __init__(self, cooldown_seconds: float = 60.0):
        self.cooldown_seconds = cooldown_seconds
        self._cooldowns: dict[str, float] = {}  # preset_id -> cooldown_until (monotonic)
        self._lock = None  # lazily created to keep import light

    def _get_lock(self):
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        return self._lock

    def record_failure(self, preset_id: str) -> None:
        """Mark `preset_id` as failed; it will be skipped for cooldown_seconds."""
        with self._get_lock():
            self._cooldowns[preset_id] = time.monotonic() + self.cooldown_seconds

    def record_success(self, preset_id: str) -> None:
        """Clear any cooldown for `preset_id` (it's healthy again)."""
        with self._get_lock():
            self._cooldowns.pop(preset_id, None)

    def is_cooled_down(self, preset_id: str) -> bool:
        """Whether `preset_id` is currently in cooldown (should be skipped)."""
        with self._get_lock():
            until = self._cooldowns.get(preset_id)
            if until is None:
                return False
            if time.monotonic() >= until:
                # expired — clean up to keep the dict from growing
                self._cooldowns.pop(preset_id, None)
                return False
            return True

    def remaining(self, preset_id: str) -> float:
        """Seconds left in cooldown for `preset_id` (0 if not cooled down)."""
        with self._get_lock():
            until = self._cooldowns.get(preset_id)
            if until is None:
                return 0.0
            rem = until - time.monotonic()
            return max(rem, 0.0)

    def snapshot(self) -> dict[str, float]:
        """Return a copy of current cooldowns (preset_id -> seconds remaining).
        For observability/debugging."""
        with self._get_lock():
            now = time.monotonic()
            return {k: max(v - now, 0.0) for k, v in self._cooldowns.items() if v > now}
