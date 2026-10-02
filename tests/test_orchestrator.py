import json
import threading
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletionMessage

from orchestrator import Orchestrator
from models import CuisineResult, FinalOutput, OrchestratorState
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
    }
    orch.breaker = None
    orch._summary_router = None

    results = [
        EvaluationResult(
            proposal=DishProposal(
                id=1, name="Dish A", cuisine="Test",
                price_sgd=5, description="test", differentiation="",
                trend_source="",
            ),
            vetoed=True, veto_reason="Deep-fried",
            trend_heat=0,
            hawker_substitutability=0, total_score=0,
            passed=False, reasoning="",
        ),
    ]

    # Fallback path (no API key in test env) still gives the generator something
    feedback = orch._synthesize_feedback(
        evaluation_results=results,
        cuisine="Test",
        round_num=1,
    )
    assert "Deep-fried" in feedback
    assert "No proposal was usable" in feedback


# ── _process_cuisine and run(), with the LLM-facing parts replaced ──

def _proposal(id_, name, name_cn="", description="grilled and served over warm rice"):
    return {
        "id": id_, "name": name, "name_cn": name_cn, "cuisine": "Japanese",
        "price_sgd": 12.0, "description": description, "description_cn": "",
        "differentiation": "", "trend_source": "", "source_urls": [],
    }


class _ScoringEvaluator:
    """Scores each dish as told (trend, hawker), 5/5 when not told, and
    records which dishes it was given."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.seen: list[str] = []

    def evaluate(self, proposals, **kwargs):
        self.seen.extend(p["name"] for p in proposals)
        evaluations = []
        for p in proposals:
            trend, hawker = self.scores.get(p["name"], (5, 5))
            evaluations.append({
                "id": p["id"], "name": p["name"], "vetoed": False,
                "scores": {"trend_heat": {"raw": trend},
                           "hawker_substitutability": {"raw": hawker}},
            })
        return evaluations, {}, ""


def _orchestrator(monkeypatch, tmp_path, batches, evaluator, max_rounds=1):
    """An Orchestrator whose generator hands out `batches` one round at a
    time. Returns it with a dict counting generator and feedback calls."""
    remaining = iter(batches)
    calls = {"generate": 0, "feedback": 0}

    class _Generator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, **kwargs):
            calls["generate"] += 1
            return next(remaining, [])

    monkeypatch.setattr("orchestrator.GeneratorAgent", _Generator)
    # The orchestrator reads rejections from ./data/feedback
    monkeypatch.chdir(tmp_path)

    orch = Orchestrator.__new__(Orchestrator)
    orch.config = {"orchestrator": {"max_rounds_per_cuisine": max_rounds}, "llm": {}}
    orch.state_callbacks = {"on_state_change": [], "on_round_complete": []}
    orch.stop_check = None
    orch.obs = None
    orch.breaker = None
    orch.evaluator = evaluator
    orch._eval_lock = threading.Lock()
    orch.sku_loader = type("Loader", (), {"get_by_cuisine": lambda self, cuisine: []})()

    def _feedback(**kwargs):
        calls["feedback"] += 1
        return "brief"

    monkeypatch.setattr(orch, "_synthesize_feedback", _feedback)
    return orch, calls


def test_hitl_blacklisted_proposal_is_not_evaluated_or_locked(tmp_path, monkeypatch):
    """A dish a human rejected stops at the blacklist. It must not reach the
    evaluator or the final recommendations, even when the guard and the
    duplicate check have nothing against it."""
    # Both dishes pass the guard and the duplicate check; only the first one
    # is on the HITL blacklist.
    proposals = [
        _proposal(1, "Miso Glazed Cod Donburi", "味噌银鳕鱼饭",
                  "oven baked cod with miso glaze over warm rice"),
        _proposal(2, "Chicken Teriyaki Rice Bowl", "照烧鸡饭",
                  "grilled chicken with teriyaki sauce over rice"),
    ]
    orch, _ = _orchestrator(monkeypatch, tmp_path, [proposals], _ScoringEvaluator())
    feedback_dir = tmp_path / "data" / "feedback"
    feedback_dir.mkdir(parents=True)
    (feedback_dir / "rejections_test.json").write_text(json.dumps({
        "rejections": [{
            "proposal_name": "Miso Glazed Cod Donburi",
            "cuisine": "Japanese",
            "reason_label": "已有同款",
        }],
    }), encoding="utf-8")

    result = orch._process_cuisine("Japanese", target=10)

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


def test_cuisine_keeps_its_best_scoring_dishes(tmp_path, monkeypatch):
    """The quota is filled with the highest dish scores, whatever order the
    dishes were proposed in. No dish has to reach a line."""
    proposals = [
        _proposal(1, "Teriyaki Chicken Don"),
        _proposal(2, "Miso Glazed Cod Donburi"),
        _proposal(3, "Yuzu Kosho Chicken Bowl"),
        _proposal(4, "Ginger Beef Rice Bowl"),
    ]
    evaluator = _ScoringEvaluator({
        "Teriyaki Chicken Don": (2, 2),
        "Miso Glazed Cod Donburi": (9, 9),
        "Yuzu Kosho Chicken Bowl": (4, 4),
        "Ginger Beef Rice Bowl": (7, 7),
    })
    orch, _ = _orchestrator(monkeypatch, tmp_path, [proposals], evaluator)

    result = orch._process_cuisine("Japanese", target=2)

    assert [e.proposal.name for e in result.locked] == [
        "Miso Glazed Cod Donburi", "Ginger Beef Rice Bowl",
    ]
    assert [e.total_score for e in result.locked] == [90.0, 70.0]
    round_result = result.rounds_history[0]
    # All four were usable candidates; two were selected.
    assert (round_result.passed_count, round_result.rejected_count) == (4, 0)


def test_another_round_runs_only_while_the_quota_is_unfilled(tmp_path, monkeypatch):
    first = [_proposal(1, "Teriyaki Chicken Don")]
    second = [_proposal(2, "Miso Glazed Cod Donburi"), _proposal(3, "Ginger Beef Rice Bowl")]
    orch, calls = _orchestrator(
        monkeypatch, tmp_path, [first, second, [_proposal(4, "Never Asked For")]],
        _ScoringEvaluator(), max_rounds=3,
    )

    result = orch._process_cuisine("Japanese", target=2)

    assert result.total_rounds == 2          # round 1 left the quota short, round 2 filled it
    assert calls["generate"] == 2
    assert calls["feedback"] == 1            # only before the round that used it
    assert len(result.locked) == 2


def test_evaluator_that_returns_nothing_shows_up_in_the_round(tmp_path, monkeypatch):
    """An evaluator outage used to look like an empty batch. Now every dish
    it failed to score is in the round, marked as such."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ([], {}, "")
    proposals = [_proposal(1, "Teriyaki Chicken Don"), _proposal(2, "Miso Glazed Cod Donburi")]
    orch, _ = _orchestrator(monkeypatch, tmp_path, [proposals], evaluator)

    result = orch._process_cuisine("Japanese", target=2)

    assert result.locked == []
    round_result = result.rounds_history[0]
    assert (round_result.passed_count, round_result.rejected_count) == (0, 2)
    assert all("未返回" in e.veto_reason for e in round_result.evaluations)


def _run_recording_targets(monkeypatch, sku_counts, **run_kwargs):
    """Call run() with _process_cuisine stubbed; return the targets it was given."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = {"orchestrator": {
        "total_target": 10,
        "cuisines": ["Mexican", "Korean", "Japanese", "Thai", "Chinese", "Singaporean/Malay"],
    }, "llm": {}}
    orch.state = OrchestratorState.IDLE
    orch.evaluator = MagicMock()
    orch.sku_loader = type("Loader", (), {"get_cuisine_counts": lambda self: sku_counts})()
    given = {}

    def _process(cuisine, target):
        given[cuisine] = target
        return CuisineResult(cuisine=cuisine, total_rounds=0, locked=[], rounds_history=[])

    monkeypatch.setattr(orch, "_process_cuisine", _process)
    monkeypatch.setattr(orch, "_generate_executive_summary", lambda output: None)

    output = orch.run(**run_kwargs)
    return given, output


def test_run_splits_the_batch_by_how_few_skus_a_cuisine_has(monkeypatch):
    """Which cuisine gets more dishes is decided here, from the SKU counts,
    and not inside any dish's score."""
    given, _ = _run_recording_targets(monkeypatch, {
        "Korean": 12, "Japanese": 14, "Thai": 14, "Chinese": 14,
        "Singaporean/Malay": 36, "Other": 47,   # Mexican: none yet
    })

    assert given == {
        "Mexican": 5, "Korean": 1, "Japanese": 1,
        "Thai": 1, "Chinese": 1, "Singaporean/Malay": 1,
    }


def test_run_takes_explicit_targets_and_skips_a_quota_of_zero(monkeypatch):
    given, output = _run_recording_targets(
        monkeypatch, {},
        cuisines=["Mexican", "Thai"], targets={"Mexican": 3, "Thai": 0},
    )

    assert given == {"Mexican": 3}
    assert list(output.cuisines) == ["Mexican"]


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
    orch.config = {"llm": {}}
    orch._summary_router = _router_returning('{"feedback": "try regional dishes"}')

    feedback = orch._synthesize_feedback(evaluation_results=[], cuisine="Test", round_num=1)

    assert feedback == "try regional dishes"
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
