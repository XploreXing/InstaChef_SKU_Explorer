"""Tests for HITL feedback system — models, persistence, self-improvement."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
import pytest
from models import RejectionFeedback
from app import REJECTION_REASONS, _save_feedback, _load_all_rejections


class TestRejectionFeedbackModel:
    def test_create_feedback(self):
        rf = RejectionFeedback(
            proposal_name="Mexican Shrimp & Mango Bowl",
            proposal_name_cn="墨西哥鲜虾芒果碗",
            cuisine="Mexican",
            reason_code="cold_food",
            reason_label="冷食/不适合60-70°C热柜",
            custom_note="芒果是冷食",
            timestamp="2026-05-29T12:00:00",
        )
        assert rf.reason_code == "cold_food"
        assert rf.cuisine == "Mexican"
        assert rf.custom_note == "芒果是冷食"

    def test_feedback_serializable(self):
        from dataclasses import asdict
        rf = RejectionFeedback(
            proposal_name="Test Dish",
            proposal_name_cn="测试菜",
            cuisine="Test",
            reason_code="duplicate",
            reason_label="已有类似SKU/重复",
        )
        d = asdict(rf)
        assert d["proposal_name"] == "Test Dish"
        assert d["reason_code"] == "duplicate"

    def test_feedback_defaults(self):
        rf = RejectionFeedback(
            proposal_name="Test",
            proposal_name_cn="测试",
            cuisine="Test",
            reason_code="other",
            reason_label="其他",
        )
        assert rf.custom_note == ""
        assert rf.timestamp == ""


class TestRejectionReasons:
    def test_all_reasons_have_code_label_and_target(self):
        for code, (label, target) in REJECTION_REASONS.items():
            assert isinstance(code, str) and len(code) > 0
            assert isinstance(label, str) and len(label) > 0
            assert target in ("guard_kill", "guard_haram", "evaluator_fried",
                              "evaluator_duplicate", "evaluator_trend",
                              "generator_naming", "evaluator_complexity",
                              "generator_price", "manual_review")

    def test_rejection_reasons_count(self):
        assert len(REJECTION_REASONS) >= 8


class TestFeedbackPersistence:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            rejections = [
                RejectionFeedback(
                    proposal_name="Dish A", proposal_name_cn="菜A",
                    cuisine="Mexican", reason_code="cold_food",
                    reason_label="冷食",
                ),
                RejectionFeedback(
                    proposal_name="Dish B", proposal_name_cn="菜B",
                    cuisine="Korean", reason_code="duplicate",
                    reason_label="重复",
                ),
            ]
            path = _save_feedback(rejections, log_dir)
            assert path.exists()

            # Load back
            loaded = _load_all_rejections(log_dir)
            assert len(loaded) == 2
            assert loaded[0]["proposal_name"] == "Dish A"
            assert loaded[1]["reason_code"] == "duplicate"

    def test_load_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = _load_all_rejections(Path(tmpdir))
            assert loaded == []

    def test_save_adoptions_keeps_the_dish(self, tmp_path):
        """An adopted dish is saved with enough of the proposal to be replayed
        later as a positively labelled eval case."""
        from app import _save_adoptions
        from models import AdoptionFeedback

        path = _save_adoptions([
            AdoptionFeedback(
                proposal_name="Dish A", proposal_name_cn="菜A", cuisine="Japanese",
                description="Grilled chicken over rice", description_cn="烤鸡肉饭",
                price_sgd=8.9, total_score=86.0, timestamp="2026-10-02T10:00:00",
            ),
        ], tmp_path)

        saved = json.loads(path.read_text())
        assert path.name.startswith("adoptions_")
        assert saved["count"] == 1
        assert saved["adoptions"][0]["proposal_name"] == "Dish A"
        assert saved["adoptions"][0]["description"] == "Grilled chicken over rice"
        # Adoptions must not be picked up as rejections by the HITL loader.
        assert _load_all_rejections(tmp_path) == []


def _locked(name, name_cn, score):
    from models import DishProposal, EvaluationResult

    return EvaluationResult(
        proposal=DishProposal(
            id=1, name=name, name_cn=name_cn, cuisine="Japanese", price_sgd=9.5,
            description=f"{name} description", description_cn=f"{name_cn}描述",
            differentiation="", trend_source="",
        ),
        vetoed=False, veto_reason=None, trend_heat=8,
        hawker_substitutability=8, total_score=score, passed=True, reasoning="",
    )


def test_recorded_decisions_keep_the_dish_for_both_outcomes(tmp_path):
    """Submitting feedback saves what was adopted as well as what was
    rejected, each with the dish itself, so both can become labelled eval
    cases."""
    from app import _record_decisions

    dish_a, dish_b = _locked("Dish A", "菜A", 86.0), _locked("Dish B", "菜B", 82.0)

    _record_decisions(
        pending=[("Dish B", "cost_high", "wagyu is too dear")],
        adopted=[dish_a],
        all_evals=[dish_a, dish_b],
        log_dir=tmp_path,
    )

    (adoptions_file,) = tmp_path.glob("adoptions_*.json")
    (adoption,) = json.loads(adoptions_file.read_text())["adoptions"]
    assert adoption["proposal_name"] == "Dish A"
    assert adoption["description"] == "Dish A description"
    assert adoption["total_score"] == 86.0
    assert adoption["timestamp"]

    (rejection,) = _load_all_rejections(tmp_path)
    assert rejection["proposal_name"] == "Dish B"
    assert rejection["reason_code"] == "cost_high"
    assert rejection["custom_note"] == "wagyu is too dear"
    assert rejection["description"] == "Dish B description"
    assert rejection["timestamp"]


def test_recording_only_adoptions_writes_no_rejection_file(tmp_path):
    from app import _record_decisions

    dish_a = _locked("Dish A", "菜A", 86.0)

    _record_decisions(pending=[], adopted=[dish_a], all_evals=[dish_a], log_dir=tmp_path)

    assert len(list(tmp_path.glob("adoptions_*.json"))) == 1
    assert list(tmp_path.glob("rejections_*.json")) == []
