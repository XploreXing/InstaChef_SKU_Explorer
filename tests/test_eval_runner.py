"""The numbers evals/run_evaluator.py reports are used to decide between
evaluator configurations, so the arithmetic behind them is tested here.
No API calls: verdicts are written by hand."""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "evals" / "run_evaluator.py"
_spec = importlib.util.spec_from_file_location("run_evaluator", _SCRIPT)
run_evaluator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_evaluator)


def _case(id_, cuisine="Japanese", label=None):
    return {"id": id_, "label": label, "dish": {"name": id_, "cuisine": cuisine}}


def _v(score):
    """A scored dish, or a vetoed one when `score` is None."""
    return {"scored": score is not None, "missing": False, "score": score or 0,
            "trend": 0, "hawker": 0, "veto_reason": None if score is not None else "veto"}


def _summary(cases, per_case):
    runs = len(next(iter(per_case.values())))
    return run_evaluator.summarize(cases, {"per_case": per_case}, runs)


def test_importing_the_script_leaves_the_evaluator_search_alone():
    import agents.evaluator as evaluator_module

    assert evaluator_module.search_web_for_eval is run_evaluator._real_search


def test_top_half_is_what_a_quota_of_half_the_batch_would_keep():
    cases = [_case("a"), _case("b"), _case("c"), _case("d"), _case("e")]
    per_case = {"a": [_v(50)], "b": [_v(90)], "c": [_v(None)], "d": [_v(70)], "e": [_v(60)]}

    # five dishes -> a quota of three; the vetoed dish can never be kept
    assert run_evaluator.top_half(cases, per_case, run=0) == {"b", "d", "e"}


def test_stability_reports_score_spread_and_dishes_that_move_in_or_out():
    cases = [_case("steady"), _case("wobbles"), _case("rival"), _case("sometimes-vetoed")]
    summary = _summary(cases, {
        "steady": [_v(90), _v(90), _v(90)],
        "wobbles": [_v(80), _v(60), _v(82)],            # kept in runs 1 and 3 only
        "rival": [_v(70), _v(70), _v(70)],              # kept in run 2 only
        "sometimes-vetoed": [_v(40), _v(None), _v(44)],
    })

    assert sorted(summary["rank_flips"]) == ["rival", "wobbles"]
    assert summary["veto_flips"] == ["sometimes-vetoed"]
    assert summary["median_spread"] == 2.0              # spreads are 0, 22, 0 and 4
    assert summary["max_spread"] == (22, "wobbles")


def test_agreement_compares_accepted_and_rejected_dishes_of_the_same_cuisine():
    cases = [
        _case("good", label="accept"), _case("bad", label="reject"),
        _case("other-cuisine", cuisine="Thai", label="reject"),
        _case("unlabelled"),
    ]
    summary = _summary(cases, {
        "good": [_v(80), _v(60)],
        "bad": [_v(70), _v(75)],                        # outscores the accepted dish in run 2
        "other-cuisine": [_v(99), _v(99)],              # another cuisine: never compared with "good"
        "unlabelled": [_v(50), _v(50)],
    })

    assert summary["pairs"] == [1, 2]


def test_rejected_dishes_that_would_still_be_kept_are_counted():
    """With no accepted dish to compare against, this is all that can be said."""
    cases = [_case("rejected-top", label="reject"), _case("rejected-bottom", label="reject"),
             _case("filler-1"), _case("filler-2")]
    summary = _summary(cases, {
        "rejected-top": [_v(95), _v(95)],
        "rejected-bottom": [_v(10), _v(10)],
        "filler-1": [_v(60), _v(60)],
        "filler-2": [_v(50), _v(50)],
    })

    assert summary["pairs"] == [0, 0]
    assert summary["rejected_kept"] == [2, 4]           # 2 rejected dishes x 2 runs
    assert summary["rejected_always_kept"] == 1


def test_missing_scores_are_counted():
    cases = [_case("a"), _case("b")]
    not_scored = {**_v(None), "missing": True}
    summary = _summary(cases, {"a": [_v(60), not_scored], "b": [_v(70), _v(70)]})

    assert summary["missing"] == 1
