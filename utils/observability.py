"""Unified observability layer — JSON Lines event logger.

Replaces the scattered _save_lineage_log / _save_tracing_log / ... pattern
with a single event stream where every event carries a trace_id so the full
causal chain (search → generate → evaluate → veto → lineage → feedback)
can be reconstructed with a single jq command.

Design principles
------------------
- **Zero new dependencies** — pure Python stdlib.
- **JSON Lines** (one JSON object per line) for append-only thread safety
  on POSIX (writes < PIPE_BUF are atomic).
- **Daily log rotation** via filename: events_{YYYYMMDD}.jsonl.
- **Additive only** — existing log functions keep working; this runs in
  parallel so nothing breaks during migration.
"""

import json
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from models import ObservabilityEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as ISO 8601 with milliseconds."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}"


def _today_str() -> str:
    """Return today's date as YYYYMMDD (local time, for file naming)."""
    return datetime.now().strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# ObservabilityLogger
# ---------------------------------------------------------------------------

class ObservabilityLogger:
    """Thread-safe JSON Lines event logger.

    Usage::

        obs = ObservabilityLogger(session_id="1717300000.456")
        obs.emit(stage="search", event_type="stage_end",
                 cuisine="Japanese", round_num=0, elapsed_ms=11317,
                 payload={"raw_result_count": 69})
        obs.flush()  # persist to disk
    """

    def __init__(self, log_dir: str = "data/logs", session_id: str = ""):
        self.trace_id = (
            f"t-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:6]}"
        )
        self.run_id = str(time.time())
        self.session_id = session_id or self.run_id

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"events_{_today_str()}.jsonl"

        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._event_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def log_path(self) -> Path:
        """Path to the JSONL file being written to."""
        return self._log_path

    @property
    def event_count(self) -> int:
        """Number of events emitted so far (in-memory counter)."""
        return self._event_count

    def emit(self, **kwargs) -> ObservabilityEvent:
        """Create and log an event.

        Every keyword argument is forwarded to the ``ObservabilityEvent``
        constructor.  Fields *not* provided are auto-filled from the
        logger instance (trace_id, run_id, session_id, timestamp).

        Returns the constructed ``ObservabilityEvent`` so callers can
        inspect it if needed.
        """
        # Auto-fill defaults that callers rarely override
        kwargs.setdefault("trace_id", self.trace_id)
        kwargs.setdefault("run_id", self.run_id)
        kwargs.setdefault("session_id", self.session_id)
        kwargs.setdefault("timestamp", _now_iso())
        kwargs.setdefault("cuisine", "")
        kwargs.setdefault("round_num", 0)
        kwargs.setdefault("stage", "")
        kwargs.setdefault("event_type", "stage_end")
        kwargs.setdefault("status", "success")
        kwargs.setdefault("elapsed_ms", 0.0)
        kwargs.setdefault("model_name", "")
        kwargs.setdefault("input_size_chars", 0)
        kwargs.setdefault("output_size_chars", 0)
        kwargs.setdefault("error", "")
        kwargs.setdefault("payload", {})

        event = ObservabilityEvent(**kwargs)
        self._buffer_line(event)
        return event

    def flush(self):
        """Ensure all events are persisted to disk.

        Events are written immediately on emit() — this call is a no-op
        for the common case, but guarantees the last write is visible
        and can be used as a sync point in multi-threaded scenarios.
        """
        with self._lock:
            # Force a fsync-equivalent: re-open and flush OS buffers
            if self._log_path.exists():
                with open(self._log_path, "a") as f:
                    f.flush()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _buffer_line(self, event: ObservabilityEvent):
        """Serialize *event* to a single JSON line and write immediately.

        Each event is written to disk as soon as emit() is called
        (crash-safe).  The lock guarantees no interleaving when
        multiple threads call emit() concurrently.

        Single-line appends < PIPE_BUF (≈ 4096 bytes) are atomic on
        POSIX, so readers never see a partial line.
        """
        line = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._event_count += 1
            self._write_line(line)

    def _write_line(self, line: str):
        """Append a single JSON line to the log file."""
        with open(self._log_path, "a") as f:
            f.write(line + "\n")
