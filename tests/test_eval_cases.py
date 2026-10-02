"""Runs the deterministic eval cases in evals/cases/ on every test run.

Each case is one dish and the verdict a deterministic layer must give it.
A case marked `known_gap` is one the system gets wrong today: it is reported
as xfail, and turns into a failure once the gap is fixed so that the marker
gets removed from the case file.
"""
from pathlib import Path

import pytest
import yaml

from utils.duplication_checker import check_duplicates
from utils.guard import HardConstraintGuard

CASES_DIR = Path(__file__).resolve().parent.parent / "evals" / "cases"

# How each rule family shows up in the guard's veto reason.
RULE_MARKERS = {
    "haram": "清真违规",
    "kill": "冷食/生食",
    "fried": "油炸",
    "hawker": "小贩中心",
}


def _load(filename: str, expectations: set[str]) -> list:
    cases = yaml.safe_load((CASES_DIR / filename).read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), f"{filename}: duplicate case ids"
    for case in cases:
        assert case["expect"] in expectations, (
            f"{filename}: {case['id']}: expect must be one of {sorted(expectations)}")
    return [
        pytest.param(
            case,
            id=case["id"],
            marks=pytest.mark.xfail(reason=case["known_gap"], strict=True)
            if case.get("known_gap") else (),
        )
        for case in cases
    ]


@pytest.mark.parametrize("case", _load("guard.yaml", {"veto", "pass"}))
def test_guard_case(case):
    passed, reason = HardConstraintGuard.precheck(case["dish"])

    if case["expect"] == "pass":
        assert passed, f"vetoed a dish that should pass: {reason}"
    else:
        assert not passed, "let through a dish that should be vetoed"
        assert RULE_MARKERS[case["rule"]] in reason, f"vetoed by the wrong rule: {reason}"


@pytest.mark.parametrize("case", _load("duplicate.yaml", {"duplicate", "distinct"}))
def test_duplicate_case(case):
    is_duplicate, reason, similarity = check_duplicates(
        name=case["proposal"]["name"],
        name_cn=case["proposal"].get("name_cn", ""),
        existing_skus=case["existing"],
    )

    assert is_duplicate == (case["expect"] == "duplicate"), (
        f"{reason or 'no match'} (best similarity {similarity:.0%})")
