"""The counting behind evals/run_feedback.py, tested without API calls, and
the case file it reads."""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "evals" / "run_feedback.py"
_spec = importlib.util.spec_from_file_location("run_feedback", _SCRIPT)
run_feedback = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_feedback)

CASES = [
    {"id": "costly", "expect": "veto"},
    {"id": "complex", "expect": "veto"},
    {"id": "plain", "expect": "keep"},
    {"id": "borderline", "expect": "unclear"},
]
REJECTIONS = [{"proposal_name": "Wagyu Bowl", "proposal_name_cn": "和牛丼"}]


def _run(costly, complex_, plain, borderline=False):
    def verdict(vetoed, reason=""):
        return {"vetoed": vetoed, "missing": False, "reason": reason if vetoed else "", "score": 70.0}
    return {
        "costly": verdict(costly, "与被拒的「和牛丼」同类，成本过高"),
        "complex": verdict(complex_, "工序太复杂"),
        "plain": verdict(plain, "too plain"),
        "borderline": verdict(borderline, "?"),
    }


def test_counts_right_vetoes_wrong_vetoes_and_fully_right_runs():
    summary = run_feedback.summarize(CASES, REJECTIONS, [
        _run(costly=True, complex_=True, plain=False),     # exactly right
        _run(costly=True, complex_=False, plain=False),    # missed one
        _run(costly=True, complex_=True, plain=True),      # vetoed an ordinary dish
    ])

    assert summary["vetoed_right"] == [5, 6]
    assert summary["vetoed_wrong"] == [1, 3]
    assert summary["fully_right"] == [1, 3]


def test_unclear_dishes_do_not_count_either_way():
    summary = run_feedback.summarize(CASES, REJECTIONS, [
        _run(costly=True, complex_=True, plain=False, borderline=True),
    ])

    assert summary["fully_right"] == [1, 1]
    assert summary["vetoed_wrong"] == [0, 1]


def test_only_vetoes_that_name_a_rejected_dish_count_as_following_it():
    summary = run_feedback.summarize(CASES, REJECTIONS, [
        _run(costly=True, complex_=True, plain=False),
    ])

    # "costly" cites 和牛丼; "complex" was vetoed without naming a precedent
    assert summary["named_precedent"] == [1, 2]


def test_run_with_an_unscored_dish_is_left_out():
    """A failed API call must not be read as the evaluator vetoing everything."""
    failed = _run(costly=True, complex_=True, plain=True)
    for verdict in failed.values():
        verdict["missing"] = True
    summary = run_feedback.summarize(CASES, REJECTIONS, [
        _run(costly=True, complex_=True, plain=False), failed,
    ])

    assert (summary["runs"], summary["complete"]) == (2, 1)
    assert summary["vetoed_wrong"] == [0, 1]


def test_case_file_is_well_formed():
    cuisine, rejections, cases = run_feedback.load()

    assert all(r["cuisine"] == cuisine for r in rejections)
    assert len({c["id"] for c in cases}) == len(cases)
    assert {c["expect"] for c in cases} <= {"veto", "keep", "unclear"}
    assert all(c.get("because") for c in cases if c["expect"] == "veto")
