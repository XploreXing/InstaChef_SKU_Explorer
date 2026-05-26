import pytest
from orchestrator import Orchestrator
from models import OrchestratorState


def test_orchestrator_init():
    orch = Orchestrator("config.yaml")
    assert orch.state == OrchestratorState.IDLE
    assert orch.config is not None
    assert orch.results == {}


def test_load_skus():
    orch = Orchestrator("config.yaml")
    orch.load_skus()
    assert orch.state == OrchestratorState.INIT
    assert orch.sku_loader is not None
    assert orch.existing_skus is not None
    assert len(orch.existing_skus) > 0


def test_synthesize_feedback_fallback():
    """Test that _synthesize_feedback fallback works without API key."""
    from models import EvaluationResult, DishProposal

    orch = Orchestrator.__new__(Orchestrator)
    orch.config = {
        "llm": {
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key_env": "LLM_API_KEY",
            "generator_model": "deepseek-ai/DeepSeek-V3",
        },
        "orchestrator": {"pass_threshold": 80},
    }

    results = [
        EvaluationResult(
            proposal=DishProposal(
                id=1, name="Dish A", cuisine="Test",
                price_sgd=5, description="test", differentiation="",
                trend_source="",
            ),
            vetoed=True, veto_reason="Deep-fried",
            cuisine_blue_ocean=0, trend_heat=0,
            hawker_substitutability=0, total_score=0,
            passed=False, reasoning="",
        ),
    ]

    # Fallback path (no API key in test env) should return text + threshold
    feedback, new_threshold = orch._synthesize_feedback(
        evaluation_results=results,
        cuisine="Test",
        pass_threshold=80,
        round_num=1,
        sku_count=5,
    )
    assert isinstance(feedback, str)
    assert len(feedback) > 0
    assert isinstance(new_threshold, int)


def test_should_continue():
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = {
        "orchestrator": {
            "target_per_cuisine": 10,
            "max_rounds_per_cuisine": 3,
        }
    }
    # Not enough locked, not at max rounds
    assert orch._should_continue(5, 2) is True
    # Enough locked
    assert orch._should_continue(10, 1) is False
    # At max rounds
    assert orch._should_continue(5, 3) is False
    # Over max rounds
    assert orch._should_continue(5, 4) is False
