import json
from agents.generator import GeneratorAgent, build_generator_user_message


def test_build_user_message_first_round():
    msg = build_generator_user_message(
        cuisine="Mexican",
        count=7,
        feedback="",
        locked_names=[],
        round_num=1,
    )
    assert "Mexican" in msg
    assert "FIRST ROUND" in msg
    assert "discover_cuisine" in msg
    assert "Generate 7" in msg


def test_build_user_message_with_feedback():
    msg = build_generator_user_message(
        cuisine="Japanese",
        count=5,
        feedback="Avoid teriyaki. Focus on curry and omurice.",
        locked_names=["Chicken Teriyaki Don", "Salmon Teriyaki Don"],
        round_num=2,
    )
    assert "Avoid teriyaki" in msg
    assert "Chicken Teriyaki Don" in msg
    assert "round 2" in msg
    assert "Generate 5" in msg


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
