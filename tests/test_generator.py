import json
from agents.generator import GeneratorAgent, build_generator_user_message


def test_build_user_message_first_round():
    msg = build_generator_user_message(
        cuisine="Mexican",
        count=7,
        locked_names=[],
        round_num=1,
    )
    assert "Mexican" in msg
    assert "FIRST ROUND" in msg
    assert "discover_cuisine" in msg
    assert "Generate 7" in msg


def test_build_user_message_with_dynamic_feedback():
    msg = build_generator_user_message(
        cuisine="Japanese",
        count=5,
        locked_names=["Chicken Teriyaki Don", "Salmon Teriyaki Don"],
        feedback="Avoid teriyaki — saturated category.",
        round_num=2,
    )
    # Dynamic per-round evaluator feedback lives in the user message (after the
    # cache-stable system prefix), so it does NOT bust the system-prompt cache.
    assert "Avoid teriyaki" in msg
    assert "Chicken Teriyaki Don" in msg
    assert "round 2" in msg
    assert "Generate 5" in msg


def test_build_user_message_round2_no_feedback_omits_line():
    msg = build_generator_user_message(
        cuisine="Japanese",
        count=5,
        locked_names=["Chicken Teriyaki Don"],
        feedback="",
        round_num=2,
    )
    assert "feedback from evaluator last round is:" not in msg
    assert "Chicken Teriyaki Don" in msg


def test_parse_generator_response_valid():
    agent = GeneratorAgent.__new__(GeneratorAgent)
    response = json.dumps({
        "proposals": [
            {
                "name": "Chicken Burrito Bowl",
                "cuisine": "Mexican",
                "price_sgd": 7.50,
                "description": "Test desc",
                "differentiation": "Test diff",
                "trend_source": "Test source",
            }
        ]
    })
    proposals = agent._parse_response(response)
    assert len(proposals) == 1
    assert proposals[0]["name"] == "Chicken Burrito Bowl"
    assert proposals[0]["price_sgd"] == 7.50


def test_parse_generator_response_invalid_json():
    agent = GeneratorAgent.__new__(GeneratorAgent)
    proposals = agent._parse_response("not valid json {{{")
    assert proposals == []
