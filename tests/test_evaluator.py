import json

import pytest
from openai.types.chat import ChatCompletionMessage

from agents.evaluator import EvaluatorAgent, build_evaluator_user_message
from utils.llm_providers.base import Choice, ModelResponse


def test_build_evaluator_message():
    from models import ProcessedCommodity

    proposals = [
        {"name": "Chicken Burrito Bowl", "cuisine": "Mexican"},
    ]
    existing_skus = [
        ProcessedCommodity(
            id=93, name="Chee Cheong Fun", description="",
            cuisine_type="Chinese",
        ),
    ]
    msg = build_evaluator_user_message(
        proposals=proposals,
        cuisine_sku_count=len(existing_skus),
        round_num=1,
        locked_count=0,
        remaining=10,
    )
    assert "Chicken Burrito Bowl" in msg
    assert "Current SKU count" in msg
    assert "Round 1" in msg


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
                    "cuisine_blue_ocean": {"raw": 8, "weighted": 32.0, "reasoning": "r"},
                    "trend_heat": {"raw": 8, "weighted": 28.0, "reasoning": "r"},
                    "hawker_substitutability": {"raw": 8, "weighted": 20.0, "reasoning": "r"},
                },
                "total_score": 80.0,
                "passed": True,
            }
        ],
        "summary": {"total_proposals": 1, "passed": 1, "rejected": 0, "rejection_reasons": []},
        "improvement_suggestions": "Good variety.",
    })
    evaluations, summary, suggestions = agent._parse_response(response)
    assert len(evaluations) == 1
    assert evaluations[0]["total_score"] == 80.0
    assert evaluations[0]["passed"] is True
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
                "scores": {
                    "cuisine_blue_ocean": {"raw": 0, "weighted": 0, "reasoning": "vetoed"},
                    "trend_heat": {"raw": 0, "weighted": 0, "reasoning": "vetoed"},
                    "hawker_substitutability": {"raw": 0, "weighted": 0, "reasoning": "vetoed"},
                },
                "total_score": 0,
                "passed": False,
            }
        ],
        "summary": {"total_proposals": 1, "passed": 0, "rejected": 1, "rejection_reasons": [
            {"id": 1, "name": "Fried Chicken Rice", "reason": "Deep-fried"}
        ]},
        "improvement_suggestions": "Avoid fried foods entirely.",
    })
    evaluations, summary, suggestions = agent._parse_response(response)
    assert len(evaluations) == 1
    assert evaluations[0]["passed"] is False
    assert evaluations[0]["vetoed"] is True


def test_parse_evaluator_response_invalid():
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    agent.cfg = {}
    evaluations, summary, suggestions = agent._parse_response("not json {{{")
    assert evaluations == []
    assert suggestions == ""


# ── evaluate(): the staged loop, driven with production-shaped responses ──

def _answer(suggestions="Good variety."):
    """A final answer that follows the output schema in evaluator_system.md."""
    return json.dumps({
        "evaluations": [
            {
                "id": 1, "name": "Test Dish", "cuisine": "Chinese",
                "vetoed": False,
                "scores": {
                    "cuisine_blue_ocean": {"raw": 8, "weighted": 32.0, "reasoning": "r"},
                    "trend_heat": {"raw": 8, "weighted": 28.0, "reasoning": "r"},
                    "hawker_substitutability": {"raw": 8, "weighted": 20.0, "reasoning": "r"},
                },
                "total_score": 80.0,
                "passed": True,
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

    def chat(self, **kwargs):
        self.calls += 1
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
    return agent.evaluate(
        proposals=[{"name": "Test Dish", "cuisine": "Chinese"}],
        cuisine_sku_count=3,
    )


def test_evaluate_returns_answer_whose_summary_is_an_object():
    """The schema defines `summary` as an object. Logging it used to raise
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
