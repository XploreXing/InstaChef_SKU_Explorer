"""Tests for lineage audit log — verifying source_ref → search URL traceability."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from models import RoundResult


class TestLineageFields:
    def test_round_result_has_ref_map(self):
        rr = RoundResult(
            cuisine="Test", round_num=1,
            proposals_generated=5, passed_count=3, rejected_count=2,
            locked_total=3, evaluations=[],
            improvement_suggestions="", elapsed_seconds=10.0,
        )
        assert isinstance(rr.ref_map, dict)
        assert rr.ref_map == {}

    def test_round_result_has_lineage_results(self):
        rr = RoundResult(
            cuisine="Test", round_num=1,
            proposals_generated=5, passed_count=3, rejected_count=2,
            locked_total=3, evaluations=[],
            improvement_suggestions="", elapsed_seconds=10.0,
        )
        assert isinstance(rr.lineage_results, list)
        assert rr.lineage_results == []

    def test_lineage_roundtrip(self):
        """Ref map and lineage results survive serialization."""
        from dataclasses import asdict

        rr = RoundResult(
            cuisine="Test", round_num=1,
            proposals_generated=3, passed_count=1, rejected_count=2,
            locked_total=1, evaluations=[],
            improvement_suggestions="", elapsed_seconds=10.0,
        )
        rr.ref_map = {"REF_01": "https://example.com/1", "REF_02": "https://example.com/2"}
        rr.lineage_results = [
            {"name": "Dish A", "source_refs": ["REF_01"], "validated": True},
            {"name": "Dish B", "source_refs": [], "validated": False},
        ]

        d = asdict(rr)
        assert d["ref_map"]["REF_01"] == "https://example.com/1"
        assert len(d["lineage_results"]) == 2
        assert d["lineage_results"][1]["validated"] is False
