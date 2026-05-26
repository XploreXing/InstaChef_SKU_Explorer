"""Tests for performance tracing instrumentation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from models import StageTrace, RoundResult


class TestStageTrace:
    def test_create_trace(self):
        t = StageTrace(
            stage="search",
            elapsed_ms=1234.5,
            model_name="tavily",
            input_size_chars=50,
            output_size_chars=2000,
        )
        assert t.stage == "search"
        assert t.elapsed_ms == 1234.5
        assert t.model_name == "tavily"
        assert t.input_size_chars == 50
        assert t.output_size_chars == 2000
        assert t.error == ""

    def test_trace_with_error(self):
        t = StageTrace(
            stage="generate",
            elapsed_ms=0,
            model_name="deepseek-v3",
            input_size_chars=500,
            output_size_chars=0,
            error="API timeout",
        )
        assert t.error == "API timeout"
        assert t.output_size_chars == 0

    def test_trace_serializable(self):
        from dataclasses import asdict
        t = StageTrace(
            stage="evaluate",
            elapsed_ms=82000.0,
            model_name="deepseek-v3",
            input_size_chars=5000,
            output_size_chars=3000,
        )
        d = asdict(t)
        assert d["stage"] == "evaluate"
        assert d["elapsed_ms"] == 82000.0


class TestRoundResultWithTraces:
    def test_round_result_has_stage_traces(self):
        rr = RoundResult(
            cuisine="Test",
            round_num=1,
            proposals_generated=5,
            passed_count=3,
            rejected_count=2,
            locked_total=3,
            evaluations=[],
            improvement_suggestions="",
            elapsed_seconds=60.0,
        )
        assert isinstance(rr.stage_traces, list)
        assert len(rr.stage_traces) == 0

    def test_round_result_append_traces(self):
        rr = RoundResult(
            cuisine="Test",
            round_num=1,
            proposals_generated=5,
            passed_count=3,
            rejected_count=2,
            locked_total=3,
            evaluations=[],
            improvement_suggestions="",
            elapsed_seconds=60.0,
        )
        rr.stage_traces.append(StageTrace(
            stage="search", elapsed_ms=500.0,
            model_name="tavily", input_size_chars=10, output_size_chars=1000,
        ))
        rr.stage_traces.append(StageTrace(
            stage="generate", elapsed_ms=45000.0,
            model_name="deepseek-v3", input_size_chars=2000, output_size_chars=1500,
        ))
        assert len(rr.stage_traces) == 2
        assert rr.stage_traces[1].stage == "generate"
