"""Tests for redesigned HITL: deterministic blacklist + semantic summary."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from utils.feedback_loader import (
    _normalize_name,
    build_blacklist,
    build_feedback_summary,
)


class TestNormalizeName:
    def test_lowercase(self):
        assert _normalize_name("Yakitori Don") == _normalize_name("yakitori don")

    def test_punctuation_removed(self):
        assert _normalize_name("Shrimp & Mango Bowl") == _normalize_name("shrimp mango bowl")

    def test_apostrophe_handled(self):
        assert _normalize_name("Lion's Head") == _normalize_name("lions head")

    def test_chinese_unchanged(self):
        """Chinese names are stored as-is but lowercased."""
        n1 = _normalize_name("墨西哥鲜虾芒果碗")
        assert "墨西哥" in n1

    def test_extra_spaces_normalized(self):
        assert _normalize_name("  Beef   Bowl  ") == "beef bowl"

    def test_order_independent_normalization(self):
        """Normalized + sorted tokens should match regardless of word order."""
        a = _normalize_name("Chicken Rice Bowl", sort_tokens=True)
        b = _normalize_name("Rice Chicken Bowl", sort_tokens=True)
        assert a == b


class TestBuildBlacklist:
    def test_normalized_names_collected(self):
        rejections = [
            {"proposal_name": "Yakitori Don", "cuisine": "Japanese"},
            {"proposal_name": "Wagyu Beef Bowl", "cuisine": "Japanese"},
            {"proposal_name": "Mango Shrimp Bowl", "cuisine": "Mexican"},
        ]
        bl = build_blacklist(rejections, cuisine="Japanese")
        assert "don yakitori" in bl  # sorted tokens
        assert "beef bowl wagyu" in bl  # sorted tokens
        assert "mango shrimp bowl" not in bl  # sorted = "bowl mango shrimp", different cuisine

    def test_different_cuisines_separate(self):
        rejections = [
            {"proposal_name": "Mango Bowl", "cuisine": "Mexican"},
        ]
        bl = build_blacklist(rejections, cuisine="Japanese")
        assert len(bl) == 0

    def test_sort_tokens_catches_reordered_words(self):
        """Order-independent matching: 'Beef Bowl' == 'Bowl Beef'"""
        rejections = [
            {"proposal_name": "Beef Rice Bowl", "cuisine": "Japanese"},
        ]
        # Both should be captured
        bl = build_blacklist(rejections, cuisine="Japanese")
        # 'beef bowl rice' sorted = 'beef bowl rice'
        assert "beef bowl rice" in bl


class TestBuildFeedbackSummary:
    def test_empty_rejections(self):
        assert build_feedback_summary([], "Japanese") == ""

    def test_single_reason_summary(self):
        rejections = [
            {"cuisine": "Japanese", "reason_label": "原料成本过高", "proposal_name": "Wagyu Bowl"},
            {"cuisine": "Japanese", "reason_label": "原料成本过高", "proposal_name": "Unagi Don"},
        ]
        summary = build_feedback_summary(rejections, "Japanese")
        assert "原料成本过高" in summary
        assert "Wagyu Bowl" in summary

    def test_only_current_cuisine_in_summary(self):
        rejections = [
            {"cuisine": "Japanese", "reason_label": "成本", "proposal_name": "Wagyu"},
            {"cuisine": "Mexican", "reason_label": "冷食", "proposal_name": "Mango"},
        ]
        summary = build_feedback_summary(rejections, "Japanese")
        assert "Wagyu" in summary
        assert "Mango" not in summary
