#!/usr/bin/env python3
"""Debug entry point for evaluator.py — set breakpoints in agents/evaluator.py then run this file.

Usage:
    python debug_evaluator.py           # full debug (needs API key)
    python debug_evaluator.py --no-api  # skip API call, only test parsing logic
"""

import sys
import json

from agents.evaluator import (
    EvaluatorAgent,
    build_evaluator_user_message,
    _load_system_prompt,
)

# ── 1. Test build_evaluator_user_message (no API needed) ──────────────
print("=" * 60)
print("1. Testing build_evaluator_user_message ...")
print("=" * 60)

proposals = [
    {
        "id": 1,
        "name": "Bulgogi Rice Bowl",
        "name_cn": "韩式烤牛肉饭",
        "cuisine": "Korean",
        "description": "Grilled marinated beef over rice with gochujang sauce",
        "description_cn": "韩式烤牛肉配米饭和辣酱",
        "differentiation": "Uses halal-certified beef, not found in hawker centres",
        "trend_source": "Google Trends: bulgogi +150% in SG",
    },
    {
        "id": 2,
        "name": "Fried Chicken Karaage",
        "name_cn": "日式炸鸡",
        "cuisine": "Japanese",
        "description": "Deep-fried chicken karaage with mayo",
        "description_cn": "日式炸鸡配蛋黄酱",
        "differentiation": "Premium free-range chicken",
        "trend_source": "TikTok trending",
    },
]

msg = build_evaluator_user_message(
    proposals=proposals,
    cuisine_sku_count=3,
    round_num=1,
    locked_count=0,
    remaining=10,
    pass_threshold=80,
    hitl_context="",
)
print(msg[:500])
print("...\n✓ build_evaluator_user_message works\n")

# ── 2. Test _parse_response (no API needed) ───────────────────────────
print("=" * 60)
print("2. Testing _parse_response ...")
print("=" * 60)

agent = EvaluatorAgent.__new__(EvaluatorAgent)
agent.cfg = {}

mock_response = json.dumps({
    "evaluations": [
        {
            "id": 1, "name": "Bulgogi Rice Bowl", "cuisine": "Korean",
            "hard_constraints": {
                "halal": {"pass": True, "note": "ok"},
                "no_fried": {"pass": True, "note": "ok"},
                "no_hawker_staple": {"pass": True, "note": "ok"},
                "no_duplicate": {"pass": True, "note": "ok"},
            },
            "vetoed": False,
            "scores": {
                "cuisine_blue_ocean": {"raw": 8, "weighted": 32.0, "reasoning": "Only 3 Korean SKUs"},
                "trend_heat": {"raw": 9, "weighted": 31.5, "reasoning": "Strong Google Trends"},
                "hawker_substitutability": {"raw": 7, "weighted": 17.5, "reasoning": "Not a common hawker item"},
            },
            "total_score": 81.0,
            "passed": True,
        }
    ],
    "summary": {"total_proposals": 1, "passed": 1, "rejected": 0, "rejection_reasons": []},
    "improvement_suggestions": "Good variety of cuisines.",
})

evaluations, summary, suggestions = agent._parse_response(mock_response)
print(f"Evaluations: {len(evaluations)}")
print(f"Summary: {json.dumps(summary, indent=2)}")
print(f"Suggestions: {suggestions}")
print("✓ _parse_response works\n")

# ── 3. Test to_evaluation_results (no API needed) ─────────────────────
print("=" * 60)
print("3. Testing to_evaluation_results ...")
print("=" * 60)

results = EvaluatorAgent.to_evaluation_results(evaluations, proposals)
for r in results:
    print(f"  {r.proposal.name}: total={r.total_score}, passed={r.passed}, vetoed={r.vetoed}")
print("✓ to_evaluation_results works\n")

# ── 4. Optional: full API call ────────────────────────────────────────
if "--no-api" not in sys.argv:
    print("=" * 60)
    print("4. Testing full evaluate() with real API call ...")
    print("=" * 60)

    import yaml
    import os

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    agent = EvaluatorAgent(config)
    evaluations, summary, suggestions = agent.evaluate(
        proposals=proposals,
        cuisine_sku_count=3,
        round_num=1,
        locked_count=0,
        remaining=10,
    )
    print(f"API returned {len(evaluations)} evaluations")
    for ev in evaluations:
        print(f"  {ev.get('name')}: total_score={ev.get('total_score')}, passed={ev.get('passed')}")
    print(f"Suggestions: {suggestions}")
    print("✓ API call works\n")
else:
    print("Skipping API call (--no-api flag set)\n")

print("All debug checks passed. Set breakpoints in agents/evaluator.py and re-run.")
