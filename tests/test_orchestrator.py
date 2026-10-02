from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletionMessage

from orchestrator import Orchestrator
from models import FinalOutput, OrchestratorState
from utils.llm_providers.base import Choice, ModelResponse


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


def _proposal(id_, name, name_cn, description):
    return {
        "id": id_, "name": name, "name_cn": name_cn, "cuisine": "Japanese",
        "price_sgd": 12.0, "description": description, "description_cn": "",
        "differentiation": "", "trend_source": "", "source_urls": [],
    }


class _RecordingEvaluator:
    """Passes every proposal it is given and records which ones it was given."""

    def __init__(self):
        self.seen: list[str] = []

    def evaluate(self, proposals, **kwargs):
        self.seen.extend(p["name"] for p in proposals)
        evaluations = [
            {"id": p["id"], "name": p["name"], "vetoed": False,
             "total_score": 90, "passed": True}
            for p in proposals
        ]
        return evaluations, {}, ""


def test_hitl_blacklisted_proposal_is_not_evaluated_or_locked(tmp_path, monkeypatch):
    """A dish a human rejected stops at the blacklist. It must not reach the
    evaluator or the final recommendations, even when the guard and the
    duplicate check have nothing against it."""
    import json
    import threading

    # Both dishes pass the guard and the duplicate check; only the first one
    # is on the HITL blacklist.
    proposals = [
        _proposal(1, "Miso Glazed Cod Donburi", "味噌银鳕鱼饭",
                  "oven baked cod with miso glaze over warm rice"),
        _proposal(2, "Chicken Teriyaki Rice Bowl", "照烧鸡饭",
                  "grilled chicken with teriyaki sauce over rice"),
    ]

    class _Generator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, **kwargs):
            return proposals

    monkeypatch.setattr("orchestrator.GeneratorAgent", _Generator)

    # The orchestrator reads rejections from ./data/feedback
    monkeypatch.chdir(tmp_path)
    feedback_dir = tmp_path / "data" / "feedback"
    feedback_dir.mkdir(parents=True)
    (feedback_dir / "rejections_test.json").write_text(json.dumps({
        "rejections": [{
            "proposal_name": "Miso Glazed Cod Donburi",
            "cuisine": "Japanese",
            "reason_label": "已有同款",
        }],
    }), encoding="utf-8")

    orch = Orchestrator.__new__(Orchestrator)
    orch.config = {
        "orchestrator": {
            "max_rounds_per_cuisine": 1,
            "target_per_cuisine": 10,
            "pass_threshold": 80,
        },
        "llm": {},
    }
    orch.state_callbacks = {"on_state_change": [], "on_round_complete": []}
    orch.stop_check = None
    orch.obs = None
    orch.breaker = None
    orch.evaluator = _RecordingEvaluator()
    orch._eval_lock = threading.Lock()
    orch.sku_loader = type("Loader", (), {"get_by_cuisine": lambda self, cuisine: []})()
    monkeypatch.setattr(orch, "_synthesize_feedback", lambda **kwargs: ("", 80))

    result = orch._process_cuisine("Japanese")

    assert orch.evaluator.seen == ["Chicken Teriyaki Rice Bowl"]
    assert [e.proposal.name for e in result.locked] == ["Chicken Teriyaki Rice Bowl"]

    round_result = result.rounds_history[0]
    blacklisted = [
        e for e in round_result.evaluations
        if e.proposal.name == "Miso Glazed Cod Donburi"
    ]
    assert len(blacklisted) == 1  # one record, not one per veto layer
    assert blacklisted[0].vetoed and "HITL" in blacklisted[0].veto_reason
    assert (round_result.passed_count, round_result.rejected_count) == (1, 1)


def _router_returning(content):
    """A stand-in LLMRouter whose chat() answers `content`, in the response
    shape the real router returns."""
    router = MagicMock()
    router.chat.return_value = ModelResponse(
        choices=[Choice(message=ChatCompletionMessage(role="assistant", content=content))],
        model="test-model", preset_id="test",
    )
    return router


def test_feedback_synthesis_runs_without_thinking():
    """max_tokens=600 leaves no room for reasoning tokens: on a model that
    thinks by default the answer came back empty and the rule-based fallback
    took over without a trace."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = {"llm": {}, "orchestrator": {"pass_threshold": 80}}
    orch._summary_router = _router_returning(
        '{"feedback": "try regional dishes", "threshold_adjustment": -5}')

    feedback, threshold = orch._synthesize_feedback(
        evaluation_results=[], cuisine="Test", pass_threshold=80,
        round_num=1, sku_count=5,
    )

    assert (feedback, threshold) == ("try regional dishes", 75)
    assert orch._summary_router.chat.call_args.kwargs["thinking"] is False


def test_executive_summary_runs_without_thinking():
    """Same budget problem as feedback synthesis, with max_tokens=500."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = {"llm": {}}
    orch._summary_router = _router_returning('{"risk_direction": "watch fried items"}')

    summary = orch._generate_executive_summary(
        FinalOutput(timestamp="t", cuisines={}, total_elapsed_seconds=0.0))

    assert summary.risk_direction == "watch fried items"
    assert orch._summary_router.chat.call_args.kwargs["thinking"] is False
