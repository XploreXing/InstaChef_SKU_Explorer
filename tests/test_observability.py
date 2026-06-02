"""Tests for utils/observability.py — unified JSON Lines event logger."""

import json
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import pytest

from models import ObservabilityEvent
from utils.observability import ObservabilityLogger, _now_iso, _today_str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_log_dir(tmp_path):
    """Provide a temporary directory for log output."""
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture
def logger(tmp_log_dir):
    """Create a fresh ObservabilityLogger pointing at tmp_log_dir."""
    return ObservabilityLogger(log_dir=str(tmp_log_dir))


# ---------------------------------------------------------------------------
# ObservabilityEvent model
# ---------------------------------------------------------------------------

class TestObservabilityEvent:
    def test_create_minimal(self):
        event = ObservabilityEvent(
            trace_id="t-test",
            run_id="r-test",
            session_id="s-test",
            timestamp="2026-06-02T10:00:00.000",
            cuisine="Japanese",
            round_num=1,
            stage="search",
            event_type="stage_end",
            status="success",
        )
        assert event.trace_id == "t-test"
        assert event.cuisine == "Japanese"
        assert event.elapsed_ms == 0.0  # default
        assert event.payload == {}       # default

    def test_create_with_payload(self):
        event = ObservabilityEvent(
            trace_id="t-test",
            run_id="r-test",
            session_id="s-test",
            timestamp="2026-06-02T10:00:00.000",
            cuisine="Japanese",
            round_num=1,
            stage="search",
            event_type="stage_end",
            status="success",
            payload={"raw_result_count": 69},
        )
        assert event.payload["raw_result_count"] == 69

    def test_default_factory_isolation(self):
        """payload field(default_factory=dict) must create independent dicts."""
        e1 = ObservabilityEvent(
            trace_id="t-1", run_id="r-1", session_id="s-1",
            timestamp="2026-06-02T10:00:00.000", cuisine="", round_num=0,
            stage="", event_type="stage_end", status="success",
        )
        e2 = ObservabilityEvent(
            trace_id="t-2", run_id="r-2", session_id="s-2",
            timestamp="2026-06-02T10:00:00.000", cuisine="", round_num=0,
            stage="", event_type="stage_end", status="success",
        )
        e1.payload["x"] = 1
        assert "x" not in e2.payload


# ---------------------------------------------------------------------------
# ObservabilityLogger — basic
# ---------------------------------------------------------------------------

class TestLoggerBasic:
    def test_emit_returns_event(self, logger):
        event = logger.emit(
            cuisine="Japanese", round_num=1, stage="search",
            event_type="stage_end",
        )
        assert isinstance(event, ObservabilityEvent)
        assert event.cuisine == "Japanese"
        assert event.round_num == 1
        assert event.stage == "search"

    def test_emit_auto_fills_trace_id(self, logger):
        event = logger.emit(
            cuisine="Korean", round_num=0, stage="summarize",
            event_type="stage_end",
        )
        assert event.trace_id == logger.trace_id
        assert event.trace_id.startswith("t-")

    def test_emit_auto_fills_timestamp(self, logger):
        event = logger.emit(
            cuisine="Thai", round_num=2, stage="generate",
            event_type="stage_end",
        )
        assert "T" in event.timestamp  # ISO 8601

    def test_trace_id_unique(self, tmp_log_dir):
        l1 = ObservabilityLogger(log_dir=str(tmp_log_dir))
        l2 = ObservabilityLogger(log_dir=str(tmp_log_dir))
        assert l1.trace_id != l2.trace_id

    def test_run_id_is_epoch_float(self, logger):
        assert "." in logger.run_id or logger.run_id.replace(".", "").isdigit()

    def test_event_count(self, logger):
        assert logger.event_count == 0
        logger.emit(cuisine="Mexican", round_num=1, stage="evaluate",
                     event_type="stage_end")
        assert logger.event_count == 1
        logger.emit(cuisine="Mexican", round_num=1, stage="evaluate",
                     event_type="veto", payload={"proposal_name": "test"})
        assert logger.event_count == 2


# ---------------------------------------------------------------------------
# Flush & file output
# ---------------------------------------------------------------------------

class TestLoggerFlush:
    def test_flush_writes_file(self, logger):
        logger.emit(
            cuisine="Chinese", round_num=1, stage="generate",
            event_type="stage_end", payload={"proposals_count": 10},
        )
        logger.flush()
        assert logger.log_path.exists()

    def test_flush_writes_valid_jsonl(self, logger):
        logger.emit(
            cuisine="Japanese", round_num=0, stage="search",
            event_type="stage_end", elapsed_ms=11317.0,
            payload={"raw_result_count": 69},
        )
        logger.flush()

        lines = logger.log_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        obj = json.loads(lines[0])
        assert obj["cuisine"] == "Japanese"
        assert obj["stage"] == "search"
        assert obj["elapsed_ms"] == 11317.0
        assert obj["payload"]["raw_result_count"] == 69

    def test_flush_multiple_events(self, logger):
        for i in range(5):
            logger.emit(
                cuisine="Korean", round_num=i + 1, stage="generate",
                event_type="stage_end",
            )
        logger.flush()

        lines = logger.log_path.read_text().strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            obj = json.loads(line)
            assert obj["round_num"] == i + 1

    def test_flush_empty_buffer(self, logger):
        """Flushing an empty buffer should not crash or create garbage."""
        logger.flush()
        # File may not exist (nothing was written) or be empty — both fine
        if logger.log_path.exists():
            assert logger.log_path.read_text().strip() == ""

    def test_events_share_trace_id(self, logger):
        logger.emit(cuisine="Thai", round_num=0, stage="search",
                     event_type="stage_end")
        logger.emit(cuisine="Thai", round_num=1, stage="generate",
                     event_type="stage_end")
        logger.emit(cuisine="Thai", round_num=1, stage="evaluate",
                     event_type="stage_end")
        logger.flush()

        lines = logger.log_path.read_text().strip().split("\n")
        trace_ids = {json.loads(l)["trace_id"] for l in lines}
        assert len(trace_ids) == 1
        assert trace_ids.pop() == logger.trace_id


# ---------------------------------------------------------------------------
# Stage-specific payloads
# ---------------------------------------------------------------------------

class TestStagePayloads:
    def test_veto_event(self, logger):
        logger.emit(
            cuisine="Japanese", round_num=1, stage="evaluate",
            event_type="veto", status="success",
            payload={
                "proposal_name": "Salmon Sashimi Bowl",
                "proposal_name_cn": "三文鱼刺身碗",
                "veto_layer": "guard",
                "veto_reason": "KILL: sashimi - cold food",
                "immediate": True,
            },
        )
        logger.flush()
        obj = json.loads(logger.log_path.read_text().strip())
        assert obj["event_type"] == "veto"
        assert obj["payload"]["veto_layer"] == "guard"

    def test_lineage_event(self, logger):
        logger.emit(
            cuisine="Chinese", round_num=1, stage="generate",
            event_type="lineage", status="success",
            payload={
                "proposal_name": "Mapo Tofu",
                "proposal_name_cn": "麻婆豆腐",
                "source_refs": ["REF_01", "REF_02"],
                "validated": True,
                "evidence_level": "menu",
                "dish_name_matched": True,
                "trend_matched": True,
                "hop2_refs": 1,
                "checked_terms": ["dish:mapo tofu"],
                "evidence_hits": [{"term": "mapo tofu", "snippet": "..."}],
            },
        )
        logger.flush()
        obj = json.loads(logger.log_path.read_text().strip())
        assert obj["event_type"] == "lineage"
        assert obj["payload"]["dish_name_matched"] is True
        assert obj["payload"]["evidence_level"] == "menu"

    def test_judging_event(self, logger):
        logger.emit(
            cuisine="Mexican", round_num=2, stage="judging",
            event_type="stage_end",
            payload={
                "evidence_penalties_applied": 3,
                "threshold": 75,
                "new_locked": 2,
                "total_locked": 7,
            },
        )
        logger.flush()
        obj = json.loads(logger.log_path.read_text().strip())
        assert obj["payload"]["threshold"] == 75
        assert obj["payload"]["total_locked"] == 7


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_emits(self, tmp_log_dir):
        """Multiple threads emitting concurrently should not corrupt output."""
        logger = ObservabilityLogger(log_dir=str(tmp_log_dir))
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(20):
                    logger.emit(
                        cuisine=f"cuisine-{thread_id}",
                        round_num=i % 3 + 1,
                        stage="generate",
                        event_type="stage_end",
                        payload={"thread": thread_id, "seq": i},
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

        logger.flush()
        lines = logger.log_path.read_text().strip().split("\n")
        # Each of 5 threads emits 20 events → 100 total
        assert len(lines) == 100

        # Every line must be valid JSON
        for line in lines:
            obj = json.loads(line)
            assert "trace_id" in obj
            assert "payload" in obj

    def test_flush_thread_safety(self, tmp_log_dir):
        """Simultaneous flush + emit should not deadlock."""
        logger = ObservabilityLogger(log_dir=str(tmp_log_dir))
        errors = []

        def emitter():
            try:
                for _ in range(10):
                    logger.emit(
                        cuisine="test", round_num=1, stage="generate",
                        event_type="stage_end",
                    )
            except Exception as e:
                errors.append(e)

        def flusher():
            try:
                for _ in range(3):
                    logger.flush()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=emitter)
        t2 = threading.Thread(target=flusher)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_now_iso_format(self):
        ts = _now_iso()
        # e.g. "2026-06-02T10:00:11.456"
        assert "T" in ts
        parts = ts.split("T")
        assert len(parts[0].split("-")) == 3  # date
        time_part = parts[1]
        assert "." in time_part  # ms
        ms_part = time_part.split(".")[1]
        assert len(ms_part) == 3

    def test_today_str_format(self):
        s = _today_str()
        assert len(s) == 8
        assert s.isdigit()


# ---------------------------------------------------------------------------
# Integration: simulate a mini pipeline run
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_mini_pipeline(self, tmp_log_dir):
        """Simulate a 2-round pipeline and verify events are traceable."""
        logger = ObservabilityLogger(log_dir=str(tmp_log_dir))

        # Round 0 — search + summarize
        logger.emit(
            cuisine="Japanese", round_num=0, stage="search",
            event_type="stage_end", elapsed_ms=5000,
            payload={"raw_result_count": 30},
        )
        logger.emit(
            cuisine="Japanese", round_num=0, stage="summarize",
            event_type="stage_end", elapsed_ms=2,
            payload={"ref_map_size": 30, "menu_refs": 5},
        )

        # Round 1
        logger.emit(
            cuisine="Japanese", round_num=1, stage="generate",
            event_type="stage_end", elapsed_ms=8000,
            payload={"proposals_count": 10},
        )
        # 2 vetoes
        logger.emit(
            cuisine="Japanese", round_num=1, stage="evaluate",
            event_type="veto",
            payload={"proposal_name": "Sashimi Bowl", "veto_layer": "guard"},
        )
        logger.emit(
            cuisine="Japanese", round_num=1, stage="generate",
            event_type="veto",
            payload={"proposal_name": "Duplicate Don", "veto_layer": "hitl_blacklist"},
        )
        # lineage for 8 passed proposals
        for i in range(8):
            logger.emit(
                cuisine="Japanese", round_num=1, stage="generate",
                event_type="lineage",
                payload={
                    "proposal_name": f"Dish_{i}",
                    "dish_name_matched": i < 3,
                    "trend_matched": True,
                },
            )
        logger.emit(
            cuisine="Japanese", round_num=1, stage="evaluate",
            event_type="stage_end", elapsed_ms=12000,
            payload={"passed_count": 3, "vetoed_count": 2, "avg_score": 68.5},
        )
        logger.emit(
            cuisine="Japanese", round_num=1, stage="judging",
            event_type="stage_end",
            payload={"evidence_penalties_applied": 2, "new_locked": 2},
        )
        logger.emit(
            cuisine="Japanese", round_num=1, stage="feedback",
            event_type="stage_end", elapsed_ms=3000,
            payload={"threshold_after": 75, "auto_adjusted": True},
        )

        # Round 2
        logger.emit(
            cuisine="Japanese", round_num=2, stage="generate",
            event_type="stage_end", elapsed_ms=7000,
            payload={"proposals_count": 8},
        )
        logger.emit(
            cuisine="Japanese", round_num=2, stage="evaluate",
            event_type="stage_end", elapsed_ms=10000,
            payload={"passed_count": 5, "avg_score": 72.0},
        )

        logger.flush()

        # Read back
        lines = logger.log_path.read_text().strip().split("\n")
        events = [json.loads(l) for l in lines]

        # All events share the same trace_id
        trace_ids = {e["trace_id"] for e in events}
        assert len(trace_ids) == 1

        # Query: all vetoes
        vetoes = [e for e in events if e["event_type"] == "veto"]
        assert len(vetoes) == 2

        # Query: all stage_end events with timing
        stage_ends = [e for e in events if e["event_type"] == "stage_end"]
        assert len(stage_ends) == 8  # search + summarize + generate*2 + evaluate*2 + judging + feedback

        # Query: lineage events for round 1
        r1_lineage = [
            e for e in events
            if e["event_type"] == "lineage" and e["round_num"] == 1
        ]
        assert len(r1_lineage) == 8

        # Query: dish_name_matched stats
        matched = sum(
            1 for e in r1_lineage
            if e["payload"].get("dish_name_matched")
        )
        assert matched == 3

        # Query: threshold was auto-adjusted
        fb_events = [e for e in events if e["stage"] == "feedback"]
        assert len(fb_events) == 1
        assert fb_events[0]["payload"]["auto_adjusted"] is True
