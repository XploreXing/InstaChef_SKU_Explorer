"""The numbers evals/run_evaluator.py reports are used to decide between
evaluator configurations, so the arithmetic behind them is tested here.
No API calls: verdicts are written by hand."""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "evals" / "run_evaluator.py"
_spec = importlib.util.spec_from_file_location("run_evaluator", _SCRIPT)
run_evaluator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_evaluator)


def _v(score, passed, vetoed=False, recomputed=None):
    return {"passed": passed, "vetoed": vetoed, "score": score, "raw": {},
            "recomputed": score if recomputed is None else recomputed, "veto_reason": None}


def _summary(cases, per_case, threshold=80):
    return run_evaluator.summarize(cases, {"per_case": per_case}, threshold)


def test_importing_the_script_leaves_the_evaluator_search_alone():
    import agents.evaluator as evaluator_module

    assert evaluator_module.search_web_for_eval is run_evaluator._real_search


def test_verdict_recomputes_the_total_from_the_dimension_scores():
    verdict = run_evaluator._verdict({
        "name": "Dish", "vetoed": False, "passed": False, "total_score": 3.25,
        "scores": {
            "cuisine_blue_ocean": {"raw": 0},
            "trend_heat": {"raw": 5},
            "hawker_substitutability": {"raw": 6},
        },
    })

    assert verdict["score"] == 3.25
    assert verdict["recomputed"] == 32.5  # 0*4 + 5*3.5 + 6*2.5


def test_agreement_counts_only_labelled_dishes():
    cases = [
        {"id": "rejected-and-caught", "label": "reject"},
        {"id": "rejected-but-passed", "label": "reject"},
        {"id": "accepted-and-passed", "label": "accept"},
        {"id": "nobody-judged-this", "label": None},
    ]
    summary = _summary(cases, {
        "rejected-and-caught": [_v(70, False), _v(72, False)],
        "rejected-but-passed": [_v(85, True), _v(75, False)],
        "accepted-and-passed": [_v(90, True), _v(88, True)],
        "nobody-judged-this": [_v(85, True), _v(85, True)],
    })

    assert summary["agree"]["reject"] == [3, 4]          # 3 of 4 verdicts were "not passed"
    assert summary["always_agree"]["reject"] == [1, 2]   # only one dish was caught every time
    assert summary["agree"]["accept"] == [2, 2]
    assert summary["always_agree"]["accept"] == [1, 1]


def test_stability_reports_spread_flips_and_missing_verdicts():
    cases = [{"id": "steady"}, {"id": "flips"}, {"id": "dropped"}]
    summary = _summary(cases, {
        "steady": [_v(71, False), _v(71, False), _v(71, False)],
        "flips": [_v(82, True), _v(77, False), _v(84, True)],
        "dropped": [_v(60, False), None, _v(64, False)],
    })

    assert summary["flips"] == ["flips"]
    assert summary["missing"] == ["dropped"]
    assert summary["median_spread"] == 4          # spreads are 0, 7 and 4
    assert summary["max_spread"] == (7, "flips")


def test_summary_flags_a_total_that_contradicts_its_own_parts():
    cases = [{"id": "bad-sum"}, {"id": "bad-flag"}, {"id": "vetoed"}]
    summary = _summary(cases, {
        "bad-sum": [_v(3.25, False, recomputed=32.5)],
        "bad-flag": [_v(72, True)],                        # passed below the threshold
        "vetoed": [_v(0, False, vetoed=True, recomputed=0)],
    })

    assert summary["miscalculated"] == [("bad-sum", 3.25, 32.5)]
    assert summary["contradictions"] == 1
