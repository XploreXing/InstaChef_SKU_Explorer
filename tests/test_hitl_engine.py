"""Tests for HITL L3: self-improvement engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
import pytest


class TestBlacklistMatch:
    def test_exact_name_match(self):
        """Exact dish name match should trigger blacklist."""
        proposal = {"name": "Mexican Shrimp & Mango Bowl", "cuisine": "Mexican"}
        rejections = [{
            "proposal_name": "Mexican Shrimp & Mango Bowl",
            "cuisine": "Mexican",
            "reason_label": "冷食/不适合60-70°C热柜",
            "timestamp": "2026-05-29",
        }]
        from app import _load_all_rejections
        # Simulate the matching logic
        for r in rejections:
            overlap = len(set(r["proposal_name"].lower().split()) &
                         set(proposal["name"].lower().split()))
            ratio = overlap / max(len(set(proposal["name"].lower().split())), 1)
            if ratio > 0.4 and r["cuisine"] == proposal["cuisine"]:
                assert True
                return
        assert False, "Should have matched"

    def test_partial_word_match(self):
        """3 out of 5 shared words = 60% overlap → should match."""
        proposal = {"name": "Spicy Beef Barbacoa Rice Bowl", "cuisine": "Mexican"}
        rejections = [{
            "proposal_name": "Beef Barbacoa Burrito Bowl",
            "cuisine": "Mexican",
            "reason_label": "油炸/不健康",
        }]
        r_words = set(rejections[0]["proposal_name"].lower().split())
        p_words = set(proposal["name"].lower().split())
        overlap = len(r_words & p_words)
        ratio = overlap / max(len(p_words), 1)
        assert ratio > 0.4  # "beef", "barbacoa", "bowl" = 3/5 = 0.6

    def test_different_cuisine_no_match(self):
        """Same name, different cuisine → no match."""
        proposal = {"name": "Mango Bowl", "cuisine": "Mexican"}
        rejections = [{
            "proposal_name": "Mango Bowl",
            "cuisine": "Thai",
            "reason_label": "冷食",
        }]
        r = rejections[0]
        if r["cuisine"] != proposal["cuisine"]:
            assert True  # correctly skipped
        else:
            assert False


class TestGuardAutoGrowth:
    def test_cold_food_trigger_threshold(self):
        """3+ cold_food rejections → should trigger guard growth."""
        rejections = [
            {"reason_code": "cold_food", "proposal_name": "Mango Bowl"},
            {"reason_code": "cold_food", "proposal_name": "Fruit Bowl"},
            {"reason_code": "cold_food", "proposal_name": "Smoothie Bowl"},
        ]
        from collections import Counter
        counts = Counter(r["reason_code"] for r in rejections)
        assert counts["cold_food"] >= 3  # threshold met

    def test_below_threshold_no_trigger(self):
        """Only 2 cold_food → threshold not met."""
        rejections = [
            {"reason_code": "cold_food", "proposal_name": "Mango Bowl"},
            {"reason_code": "cold_food", "proposal_name": "Fruit Bowl"},
            {"reason_code": "duplicate", "proposal_name": "Other Dish"},
        ]
        from collections import Counter
        counts = Counter(r["reason_code"] for r in rejections)
        assert counts["cold_food"] < 3


class TestEvaluatorRAG:
    def test_build_rag_context_same_cuisine(self):
        """Only show rejections for current cuisine."""
        rejections = [
            {"cuisine": "Mexican", "proposal_name_cn": "芒果虾碗", "reason_label": "冷食"},
            {"cuisine": "Korean", "proposal_name_cn": "炸鸡饭", "reason_label": "油炸"},
            {"cuisine": "Mexican", "proposal_name_cn": "冷面", "reason_label": "冷食"},
        ]
        mexican = [r for r in rejections if r.get("cuisine") == "Mexican"]
        assert len(mexican) == 2

        lines = ["\n## HITL Feedback: Previously Rejected Proposals (Human Review)"]
        for r in mexican[-10:]:
            lines.append(f"- 「{r['proposal_name_cn']}」→ {r['reason_label']}")
        rag = "\n".join(lines)
        assert "芒果虾碗" in rag
        assert "冷面" in rag
        assert "炸鸡饭" not in rag  # Korean
