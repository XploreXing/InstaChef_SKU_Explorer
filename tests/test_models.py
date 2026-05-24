import pytest
from models import DishProposal, EvaluationResult, RoundResult, CuisineResult, FinalOutput, OrchestratorState


def test_dish_proposal_creation():
    p = DishProposal(
        id=1,
        name="Chicken Burrito Bowl",
        cuisine="墨西哥",
        price_sgd=7.50,
        description="Test description",
        differentiation="Test differentiation",
        trend_source="Test source",
    )
    assert p.id == 1
    assert p.name == "Chicken Burrito Bowl"
    assert p.cuisine == "墨西哥"


def test_evaluation_result_passed():
    p = DishProposal(1, "Test", "中式", 5.0, "desc", "diff", "source")
    e = EvaluationResult(
        proposal=p,
        vetoed=False,
        veto_reason=None,
        cuisine_blue_ocean=9.0,
        trend_heat=8.0,
        hawker_substitutability=9.0,
        total_score=90.5,
        passed=True,
        reasoning="Test reasoning",
    )
    assert e.passed is True
    assert e.total_score == 90.5


def test_evaluation_result_vetoed():
    p = DishProposal(2, "Fried Chicken", "中式", 5.0, "desc", "diff", "source")
    e = EvaluationResult(
        proposal=p,
        vetoed=True,
        veto_reason="Deep-fried",
        cuisine_blue_ocean=0.0,
        trend_heat=0.0,
        hawker_substitutability=0.0,
        total_score=0.0,
        passed=False,
        reasoning="Hard constraint violation: deep-fried",
    )
    assert e.passed is False
    assert e.total_score == 0.0


def test_round_result():
    rr = RoundResult(
        cuisine="中式",
        round_num=1,
        proposals_generated=15,
        passed_count=4,
        rejected_count=11,
        locked_total=4,
        evaluations=[],
        improvement_suggestions="Avoid teriyaki flavor profile",
        elapsed_seconds=12.5,
    )
    assert rr.round_num == 1
    assert rr.locked_total == 4


def test_cuisine_result():
    cr = CuisineResult(
        cuisine="中式",
        total_rounds=2,
        locked=[],
        rounds_history=[],
    )
    assert cr.cuisine == "中式"
    assert cr.total_rounds == 2


def test_final_output():
    fo = FinalOutput(
        timestamp="2026-05-24T10:00:00",
        cuisines={},
        total_elapsed_seconds=120.0,
    )
    assert fo.timestamp == "2026-05-24T10:00:00"


def test_orchestrator_state_enum():
    assert OrchestratorState.IDLE.value == "idle"
    assert OrchestratorState.DONE.value == "done"
    assert len(list(OrchestratorState)) == 8
