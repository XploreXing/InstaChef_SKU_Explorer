import json
import tempfile
import os
from models import DishProposal, EvaluationResult, FinalOutput, CuisineResult
from utils.aggregator import Aggregator


def _make_proposal(id, name, cuisine):
    return DishProposal(id, name, cuisine, 5.0, "desc", "diff", "source")


def _make_eval(proposal, score, passed=True):
    return EvaluationResult(
        proposal=proposal, vetoed=not passed,
        veto_reason=None if passed else "test veto",
        trend_heat=score/35*10,
        hawker_substitutability=score/25*10,
        total_score=score, passed=passed, reasoning="test",
    )


def test_deduplicate_same_name():
    p1 = _make_proposal(1, "Chicken Burrito Bowl", "Mexican")
    p2 = _make_proposal(2, "Chicken Burrito Bowl", "Mexican")
    e1 = _make_eval(p1, 90)
    e2 = _make_eval(p2, 85)
    result = Aggregator.deduplicate([e1, e2])
    assert len(result) == 1


def test_deduplicate_different_name_keeps_both():
    p1 = _make_proposal(1, "Chicken Burrito Bowl", "Mexican")
    p2 = _make_proposal(2, "Beef Burrito Bowl", "Mexican")
    e1 = _make_eval(p1, 90)
    e2 = _make_eval(p2, 85)
    result = Aggregator.deduplicate([e1, e2])
    assert len(result) == 2


def test_sort_by_score_desc():
    p1 = _make_proposal(1, "A", "Chinese")
    p2 = _make_proposal(2, "B", "Chinese")
    p3 = _make_proposal(3, "C", "Chinese")
    evals = [
        _make_eval(p2, 82),
        _make_eval(p1, 95),
        _make_eval(p3, 88),
    ]
    sorted_evals = Aggregator.sort_by_score(evals)
    assert sorted_evals[0].total_score == 95
    assert sorted_evals[1].total_score == 88
    assert sorted_evals[2].total_score == 82


def test_write_output():
    config = {"output": {"base_dir": tempfile.mkdtemp(), "filename_prefix": "test", "format": "json"}}
    agg = Aggregator(config)
    output = FinalOutput(timestamp="2026-05-24T10:00:00", cuisines={}, total_elapsed_seconds=10.0)
    filepath = agg.write_output(output)
    assert os.path.exists(filepath)
    with open(filepath) as f:
        data = json.load(f)
    assert data["timestamp"] == "2026-05-24T10:00:00"
    os.unlink(filepath)
