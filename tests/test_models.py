import pytest
from models import (
    DishProposal, EvaluationResult, RoundResult, CuisineResult,
    FinalOutput, OrchestratorState, RawCommodity, ProcessedCommodity,
)


def test_dish_proposal_creation():
    p = DishProposal(
        id=1,
        name="Chicken Burrito Bowl",
        cuisine="Mexican",
        price_sgd=7.50,
        description="Test description",
        differentiation="Test differentiation",
        trend_source="Test source",
    )
    assert p.id == 1
    assert p.name == "Chicken Burrito Bowl"
    assert p.cuisine == "Mexican"


def test_evaluation_result_passed():
    p = DishProposal(1, "Test", "Chinese", 5.0, "desc", "diff", "source")
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
    p = DishProposal(2, "Fried Chicken", "Chinese", 5.0, "desc", "diff", "source")
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
        cuisine="Chinese",
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
        cuisine="Chinese",
        total_rounds=2,
        locked=[],
        rounds_history=[],
    )
    assert cr.cuisine == "Chinese"
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


def test_raw_commodity_from_csv():
    c = RawCommodity(id=94, name="Kimchi Fried Rice", description="Bold Korean flavors")
    assert c.id == 94
    assert c.name == "Kimchi Fried Rice"
    assert "Korean" in c.description


def test_processed_commodity():
    c = ProcessedCommodity(
        id=94, name="Kimchi Fried Rice", description="Bold Korean flavors",
        cuisine_type="Korean",
    )
    assert c.cuisine_type == "Korean"
