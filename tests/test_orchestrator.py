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


def test_extract_improvement_feedback():
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = {
        "orchestrator": {"pass_threshold": 80}
    }
    rejection_reasons = [
        {"id": 1, "name": "Dish A", "reason": "Deep-fried"},
        {"id": 2, "name": "Dish B", "reason": "Too similar to Teriyaki Chicken"},
    ]
    suggestions = "Focus on Mexican and Korean stews. Avoid teriyaki."
    feedback = orch._extract_improvement_feedback(rejection_reasons, suggestions)
    assert "Deep-fried" in feedback
    assert "Teriyaki Chicken" in feedback
    assert "Mexican" in feedback


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
