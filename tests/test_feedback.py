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
