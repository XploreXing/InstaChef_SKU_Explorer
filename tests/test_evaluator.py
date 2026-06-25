import json
from agents.evaluator import EvaluatorAgent, build_evaluator_user_message


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
