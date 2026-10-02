import json

import httpx
import pytest
from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessage

from agents.evaluator import EvaluatorAgent, build_evaluator_user_message
from utils.llm_providers.base import Choice, ModelResponse


def test_build_evaluator_message():
    msg = build_evaluator_user_message(
        [{"name": "Chicken Burrito Bowl", "cuisine": "Mexican"}],
        hitl_context="## HITL Feedback: Previously Rejected Proposals",
    )

    assert "Chicken Burrito Bowl" in msg
    assert "HITL Feedback" in msg
    # The evaluator judges dishes. It is told nothing about the run: not how
    # many SKUs the cuisine has, how many dishes are wanted, or any pass line.
    for run_detail in ("SKU", "Round", "threshold", "accepted"):
        assert run_detail not in msg


def test_parse_evaluator_response_valid():
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    agent.cfg = {}
    response = json.dumps({
        "evaluations": [
            {
                "id": 1, "name": "Test Dish", "cuisine": "Chinese",
                "hard_constraints": {
                    "halal": {"pass": True, "note": "ok"},
                    "no_fried": {"pass": True, "note": "ok"},
                    "no_hawker_staple": {"pass": True, "note": "ok"},
                    "no_duplicate": {"pass": True, "note": "ok"},
                },
                "vetoed": False,
                "scores": {
                    "trend_heat": {"raw": 8, "reasoning": "r"},
                    "hawker_substitutability": {"raw": 8, "reasoning": "r"},
                },
            }
        ],
        "improvement_suggestions": "Good variety.",
    })
    evaluations, summary, suggestions = agent._parse_response(response)
    assert len(evaluations) == 1
    assert evaluations[0]["scores"]["trend_heat"]["raw"] == 8
    assert summary == {}
    assert suggestions == "Good variety."


def test_parse_evaluator_response_vetoed():
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    agent.cfg = {}
    response = json.dumps({
        "evaluations": [
            {
                "id": 1, "name": "Fried Chicken Rice", "cuisine": "Korean",
                "hard_constraints": {
                    "halal": {"pass": True, "note": "ok"},
                    "no_fried": {"pass": False, "note": "Deep-fried chicken"},
                    "no_hawker_staple": {"pass": True, "note": "ok"},
                    "no_duplicate": {"pass": True, "note": "ok"},
                },
                "vetoed": True,
                "veto_reason": "Deep-fried",
                "scores": {
                    "trend_heat": {"raw": 0, "reasoning": "vetoed"},
                    "hawker_substitutability": {"raw": 0, "reasoning": "vetoed"},
                },
            }
        ],
        "improvement_suggestions": "Avoid fried foods entirely.",
    })
    evaluations, summary, suggestions = agent._parse_response(response)
    assert len(evaluations) == 1
    assert evaluations[0]["vetoed"] is True
    assert evaluations[0]["veto_reason"] == "Deep-fried"


def test_parse_evaluator_response_invalid():
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    agent.cfg = {}
    evaluations, summary, suggestions = agent._parse_response("not json {{{")
    assert evaluations == []
    assert suggestions == ""


# ── evaluate(): the staged loop, driven with production-shaped responses ──

def _answer(suggestions="Good variety."):
    """A final answer that follows the output schema in evaluator_system.md,
    plus a `summary` object: the schema no longer asks for one, but a model
    may still send it."""
    return json.dumps({
        "evaluations": [
            {
                "id": 1, "name": "Test Dish", "cuisine": "Chinese",
                "vetoed": False,
                "scores": {
                    "trend_heat": {"raw": 8, "reasoning": "r"},
                    "hawker_substitutability": {"raw": 8, "reasoning": "r"},
                },
            }
        ],
        "summary": {"total_proposals": 1, "passed": 1, "rejected": 0, "rejection_reasons": []},
        "improvement_suggestions": suggestions,
    })


def _router_response(content=None, tool_calls=None, finish_reason="stop"):
    """The shape LLMRouter.chat() returns in production: a ModelResponse whose
    message is the SDK's pydantic ChatCompletionMessage, not a dict or a mock."""
    message = ChatCompletionMessage.model_validate(
        {"role": "assistant", "content": content, "tool_calls": tool_calls}
    )
    return ModelResponse(
        choices=[Choice(message=message, finish_reason=finish_reason)],
        model="test-model",
        preset_id="test",
    )


class _ScriptedClient:
    """Stands in for LLMRouter: replays the scripted responses in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.kwargs: list[dict] = []  # what each chat() call was given

    def chat(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _make_agent(script):
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    agent.cfg = {"evaluator_temperature": 0.1}
    agent.system_prompt = "sys"
    agent.client = _ScriptedClient(script)
    return agent


def _evaluate(agent):
    return agent.evaluate(proposals=[{"name": "Test Dish", "cuisine": "Chinese"}])


def test_evaluate_returns_answer_whose_summary_is_an_object():
    """A model may return `summary` as an object. Logging it used to raise
    after a successful parse, which dropped the whole batch."""
    agent = _make_agent([_router_response(content=_answer())])

    evaluations, summary, suggestions = _evaluate(agent)

    assert [e["name"] for e in evaluations] == ["Test Dish"]
    assert summary["passed"] == 1
    assert suggestions == "Good variety."
    assert agent.client.calls == 1  # answered in stage 1, no fallback calls


@pytest.mark.parametrize("suggestions", [
    ["Try regional dishes", "Avoid bowls"],
    {"focus": "regional dishes"},
])
def test_evaluate_returns_answer_with_non_string_suggestions(suggestions):
    """Models often return a list or object where the schema asks for a string."""
    agent = _make_agent([_router_response(content=_answer(suggestions))])

    evaluations, _, returned = _evaluate(agent)

    assert [e["name"] for e in evaluations] == ["Test Dish"]
    assert returned == suggestions


def test_evaluate_salvages_answer_when_stage1_raises_after_answering(monkeypatch):
    """If stage 1 raises once the model's answer is already in the history,
    stage 2 re-reads it at no extra API cost."""
    agent = _make_agent([])

    def _crash_after_answer(messages, handlers):
        messages.append(_router_response(content=_answer()).choices[0].message)
        raise RuntimeError("bug between parse and return")

    monkeypatch.setattr(agent, "_run_tool_loop", _crash_after_answer)

    evaluations, _, _ = _evaluate(agent)

    assert [e["name"] for e in evaluations] == ["Test Dish"]
    assert agent.client.calls == 0


def test_evaluate_skips_stage3_when_stage1_api_call_fails():
    """Stage 3 re-sends the same history, so after a failed API call it would
    only repeat the failure, and its timeout, twice more."""
    timeout = APITimeoutError(request=httpx.Request("POST", "https://llm.invalid"))
    agent = _make_agent([timeout, _router_response(content=_answer())])

    assert _evaluate(agent) == ([], {}, "")
    assert agent.client.calls == 1


def test_tool_loop_asks_the_model_to_think():
    """Scoring is the one place we want the model to reason before answering."""
    agent = _make_agent([_router_response(content=_answer())])

    _evaluate(agent)

    assert agent.client.kwargs[0]["thinking"] is True


def test_forced_final_output_asks_the_model_to_think():
    agent = _make_agent([_router_response(content=_answer())])

    agent._force_final_output([{"role": "user", "content": "evaluate"}])

    assert agent.client.kwargs[0]["thinking"] is True


# ── to_evaluation_results(): what the model said, turned into results ──

PROPOSALS = [
    {"id": 1, "name": "Dish A", "name_cn": "菜A", "cuisine": "Japanese",
     "price_sgd": 8.0, "description": "the first dish"},
    {"id": 2, "name": "Dish B", "name_cn": "菜B", "cuisine": "Japanese",
     "price_sgd": 9.0, "description": "the second dish"},
]


def _scored(id_, name, trend, hawker, **extra):
    return {
        "id": id_, "name": name, "vetoed": False,
        "scores": {"trend_heat": {"raw": trend}, "hawker_substitutability": {"raw": hawker}},
        **extra,
    }


def test_total_score_is_computed_in_code():
    """The model's own total and verdict are ignored. It has reported totals
    on the wrong scale (3.25 for 32.5) and passed dishes under the line."""
    (result,) = EvaluatorAgent.to_evaluation_results(
        [_scored(1, "Dish A", trend=8, hawker=10, total_score=3.25, passed=False)],
        PROPOSALS[:1],
    )

    assert result.total_score == 88.3  # (8 * 3.5 + 10 * 2.5) / 60 * 100
    assert result.passed is True       # not vetoed, so still a candidate


def test_evaluations_are_matched_to_proposals_by_name_not_position():
    """Pairing by position attached one dish's scores to another dish's price
    and description whenever the model reordered or dropped a dish."""
    results = EvaluatorAgent.to_evaluation_results(
        [_scored(2, "Dish B", trend=5, hawker=5), _scored(1, "Dish A", trend=9, hawker=9)],
        PROPOSALS,
    )

    assert [r.proposal.name for r in results] == ["Dish A", "Dish B"]
    dish_a, dish_b = results
    assert (dish_a.trend_heat, dish_a.proposal.price_sgd) == (9, 8.0)
    assert (dish_b.trend_heat, dish_b.proposal.price_sgd) == (5, 9.0)


def test_renamed_dish_is_matched_by_id():
    (result,) = EvaluatorAgent.to_evaluation_results(
        [_scored(1, "Dish A (grilled)", trend=7, hawker=7)], PROPOSALS[:1],
    )

    assert result.proposal.name == "Dish A"
    assert result.trend_heat == 7


def test_veto_reason_is_read_from_the_evaluation():
    (result,) = EvaluatorAgent.to_evaluation_results(
        [{"id": 1, "name": "Dish A", "vetoed": True, "veto_reason": "搜索验证失败",
          "scores": {"trend_heat": {"raw": 9}}}],
        PROPOSALS[:1],
    )

    assert result.vetoed and not result.passed
    assert result.veto_reason == "搜索验证失败"
    assert result.total_score == 0


def test_dish_the_evaluator_skipped_is_kept_out_of_the_ranking():
    """A dish with no scores cannot be ranked. It stays in the round, marked,
    instead of vanishing: an evaluator that returns nothing must be visible."""
    results = EvaluatorAgent.to_evaluation_results([_scored(1, "Dish A", 8, 8)], PROPOSALS)

    skipped = results[1]
    assert skipped.proposal.name == "Dish B"
    assert skipped.vetoed and not skipped.passed
    assert "未返回" in skipped.veto_reason


@pytest.mark.parametrize("raw, expected", [
    (28.0, 10.0),      # a weighted value sent where a 0-10 score belongs
    (-3, 0.0),
    ("8", 0.0),        # not a number
    (None, 0.0),
])
def test_out_of_range_dimension_scores_are_clamped(raw, expected):
    (result,) = EvaluatorAgent.to_evaluation_results(
        [_scored(1, "Dish A", trend=raw, hawker=0)], PROPOSALS[:1],
    )

    assert result.trend_heat == expected
