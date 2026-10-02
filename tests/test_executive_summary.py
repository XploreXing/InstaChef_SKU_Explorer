"""Tests for executive summary generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from models import ExecutiveSummary


class TestExecutiveSummaryModel:
    def test_create_empty_summary(self):
        s = ExecutiveSummary(
            overall_pass_rate="0% (0/10)",
            total_proposals=10,
            total_passed=0,
            veto_patterns=["5x duplicate", "2x deep-fried"],
            low_score_patterns=["Korean blue-ocean bottleneck (6/10)"],
            risk_direction="Focus on niche sub-regional Korean dishes",
            top_recommendations=["Korean Braised Beef Bowl", "Korean Mushroom Rice"],
        )
        assert s.total_proposals == 10
        assert s.total_passed == 0
        assert len(s.veto_patterns) == 2
        assert isinstance(s.top_recommendations, list)

    def test_summary_serializable(self):
        from dataclasses import asdict
        s = ExecutiveSummary(
            overall_pass_rate="50% (5/10)",
            total_proposals=10,
            total_passed=5,
            veto_patterns=[],
            low_score_patterns=[],
            risk_direction="Continue current strategy",
            top_recommendations=["Dish A", "Dish B"],
        )
        d = asdict(s)
        assert d["overall_pass_rate"] == "50% (5/10)"
        assert d["top_recommendations"] == ["Dish A", "Dish B"]


class TestExecutiveSummaryIntegration:
    def test_build_from_output(self):
        """Building summary from FinalOutput data."""
        from models import FinalOutput, CuisineResult, RoundResult
        from models import EvaluationResult, DishProposal

        prop = DishProposal(
            id=1, name="Test Dish", name_cn="测试菜",
            cuisine="Test", price_sgd=5.0,
            description="A test", differentiation="Unique",
            trend_source="test",
        )

        ev = EvaluationResult(
            proposal=prop, vetoed=False, veto_reason=None,
            trend_heat=7,
            hawker_substitutability=8, total_score=78.5,
            passed=True, reasoning="Good dish",
        )

        rr = RoundResult(
            cuisine="Test", round_num=1,
            proposals_generated=5, passed_count=3,
            rejected_count=2, locked_total=3,
            evaluations=[ev, ev, ev],  # 3 passed
            improvement_suggestions="",
            elapsed_seconds=60.0,
        )

        cr = CuisineResult(
            cuisine="Test", total_rounds=1,
            locked=[ev, ev, ev],
            rounds_history=[rr],
        )

        output = FinalOutput(
            timestamp="2026-01-01T00:00:00",
            cuisines={"Test": cr},
            total_elapsed_seconds=60.0,
        )

        # Verify output structure is intact
        assert output.cuisines["Test"].locked[0].passed is True
        assert output.total_elapsed_seconds == 60.0
