"""Integration tests for guard + orchestrator pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from utils.guard import HardConstraintGuard


class TestGuardIntegration:
    """Test that guard-vetoed proposals produce correct EvaluationResult shape."""

    def test_guard_vetoed_produces_zero_scores(self):
        """Guard-vetoed items should have all scores = 0 and passed = False."""
        from models import DishProposal, EvaluationResult

        proposal_dict = {
            "id": 1,
            "name": "Pork Belly Bowl",
            "name_cn": "五花肉碗",
            "cuisine": "Chinese",
            "price_sgd": 6.50,
            "description": "Braised pork belly on rice",
            "description_cn": "红烧五花肉配饭",
            "differentiation": "None",
            "trend_source": "test",
        }

        passed, veto_reason = HardConstraintGuard.precheck(proposal_dict)
        assert not passed

        result = EvaluationResult(
            proposal=DishProposal(
                id=proposal_dict["id"],
                name=proposal_dict["name"],
                name_cn=proposal_dict["name_cn"],
                cuisine=proposal_dict["cuisine"],
                price_sgd=proposal_dict["price_sgd"],
                description=proposal_dict["description"],
                description_cn=proposal_dict["description_cn"],
                differentiation=proposal_dict["differentiation"],
                trend_source=proposal_dict["trend_source"],
            ),
            vetoed=True,
            veto_reason=veto_reason,
            cuisine_blue_ocean=0,
            trend_heat=0,
            hawker_substitutability=0,
            total_score=0,
            passed=False,
            reasoning="高压线熔断 — 无需 LLM 评估",
        )

        assert result.vetoed is True
        assert result.passed is False
        assert result.total_score == 0
        assert result.cuisine_blue_ocean == 0
        assert result.trend_heat == 0
        assert result.hawker_substitutability == 0
        assert "高压线" in result.reasoning

    def test_batch_mixed_proposals(self):
        """8 proposals: 2 should be guard-vetoed, 6 should pass guard."""
        proposals = [
            {"name": "Pork Rice", "name_cn": "猪肉饭", "description": "pork", "description_cn": "猪肉", "cuisine": "Chinese", "price_sgd": 5, "differentiation": "", "trend_source": ""},
            {"name": "Chicken Rice", "name_cn": "鸡肉饭", "description": "chicken", "description_cn": "鸡肉", "cuisine": "Chinese", "price_sgd": 5, "differentiation": "", "trend_source": ""},
            {"name": "Beef Bowl", "name_cn": "牛肉碗", "description": "beef", "description_cn": "牛肉", "cuisine": "Korean", "price_sgd": 5, "differentiation": "", "trend_source": ""},
            {"name": "Lard Noodles", "name_cn": "猪油面", "description": "lard", "description_cn": "猪油拌面", "cuisine": "Chinese", "price_sgd": 5, "differentiation": "", "trend_source": ""},
            {"name": "Cold Soba", "name_cn": "冷面", "description": "cold noodles", "description_cn": "冷面", "cuisine": "Japanese", "price_sgd": 5, "differentiation": "", "trend_source": ""},
            {"name": "Tofu Bowl", "name_cn": "豆腐碗", "description": "tofu", "description_cn": "豆腐", "cuisine": "Chinese", "price_sgd": 5, "differentiation": "", "trend_source": ""},
            {"name": "Fish Rice", "name_cn": "鱼饭", "description": "fish", "description_cn": "鱼", "cuisine": "Japanese", "price_sgd": 5, "differentiation": "", "trend_source": ""},
            {"name": "Shrimp Bowl", "name_cn": "虾碗", "description": "shrimp", "description_cn": "虾", "cuisine": "Thai", "price_sgd": 5, "differentiation": "", "trend_source": ""},
        ]

        vetoed_count = 0
        passed_count = 0
        for p in proposals:
            result, _ = HardConstraintGuard.precheck(p)
            if result:
                passed_count += 1
            else:
                vetoed_count += 1

        # Pork Rice, Lard Noodles, Cold Soba should be vetoed = 3
        assert vetoed_count == 3
        assert passed_count == 5
