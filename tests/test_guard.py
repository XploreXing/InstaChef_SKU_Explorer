"""Tests for utils/guard.py — hard-coded high-voltage guard rails."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from utils.guard import HardConstraintGuard


class TestHaramCheck:
    def test_pork_in_name_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Pork Belly Rice Bowl",
            "name_cn": "五花肉饭",
            "description": "Braised pork belly with rice",
            "description_cn": "红烧五花肉配米饭",
        })
        assert not passed
        assert "清真" in reason or "pork" in reason.lower()

    def test_alcohol_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Beer Braised Chicken",
            "name_cn": "啤酒炖鸡",
            "description": "Chicken braised in beer",
            "description_cn": "用啤酒炖的鸡肉",
        })
        assert not passed
        assert "alcohol" in reason.lower() or "酒" in reason

    def test_lard_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Lard Fried Rice",
            "name_cn": "猪油炒饭",
            "description": "Fried rice cooked with lard",
            "description_cn": "猪油炒饭",
        })
        assert not passed
        assert "lard" in reason.lower() or "猪油" in reason

    def test_halal_chicken_passes(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Grilled Chicken Rice Bowl",
            "name_cn": "烤鸡肉饭",
            "description": "Grilled halal chicken with jasmine rice",
            "description_cn": "清真烤鸡配茉莉花饭",
        })
        assert passed

    def test_beef_passes(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Beef Bulgogi Bowl",
            "name_cn": "牛肉拌饭",
            "description": "Korean style beef bulgogi with rice",
            "description_cn": "韩式牛肉拌饭",
        })
        assert passed

    def test_ham_in_name_cn_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Cheese Melt Rice",
            "name_cn": "火腿芝士焗饭",
            "description": "Cheese melt with rice",
            "description_cn": "芝士焗饭",
        })
        assert not passed

    def test_bacon_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Bacon Carbonara Rice",
            "name_cn": "培根奶油饭",
            "description": "Creamy rice with bacon",
            "description_cn": "培根奶油酱饭",
        })
        assert not passed


class TestKillCheck:
    def test_salad_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Caesar Salad Bowl",
            "name_cn": "凯撒沙拉碗",
            "description": "Cold caesar salad",
            "description_cn": "冷凯撒沙拉",
        })
        assert not passed
        assert "冷" in reason or "salad" in reason.lower()

    def test_cold_noodles_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Cold Noodle Bowl",
            "name_cn": "冷面",
            "description": "Korean cold noodles",
            "description_cn": "韩式冷面",
        })
        assert not passed

    def test_cold_brew_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Cold Brew Coffee",
            "name_cn": "冷萃咖啡",
            "description": "serve cold",
            "description_cn": "冷饮",
        })
        assert not passed

    def test_sashimi_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Salmon Sashimi Bowl",
            "name_cn": "三文鱼刺身饭",
            "description": "Fresh salmon sashimi on rice",
            "description_cn": "新鲜三文鱼刺身盖饭",
        })
        assert not passed

    def test_hot_rice_bowl_passes(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Hot Stone Bibimbap",
            "name_cn": "石锅拌饭",
            "description": "Sizzling hot stone bowl with rice",
            "description_cn": "热气腾腾的石锅拌饭",
        })
        assert passed

    def test_warm_salad_passes(self):
        """Warm salads should pass — the context of 'warm' overrides 'salad'."""
        passed, reason = HardConstraintGuard.precheck({
            "name": "Warm Potato Salad",
            "name_cn": "温土豆沙拉",
            "description": "Warm roasted potato salad",
            "description_cn": "温热烤土豆沙拉",
        })
        assert passed

    def test_ice_cream_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Ice Cream Mochi",
            "name_cn": "冰淇淋麻薯",
            "description": "Cold ice cream dessert",
            "description_cn": "冷冰淇淋甜点",
        })
        assert not passed


    def test_mango_bowl_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Mexican Shrimp & Mango Bowl",
            "name_cn": "墨西哥鲜虾芒果碗",
            "description": "Fresh mango and shrimp on rice",
            "description_cn": "新鲜芒果和虾配饭",
        })
        assert not passed
        assert "芒果" in reason or "mango" in reason.lower()

    def test_smoothie_bowl_triggers_veto(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Acai Smoothie Bowl",
            "name_cn": "果昔碗",
            "description": "Cold smoothie bowl",
            "description_cn": "冷果昔碗",
        })
        assert not passed


class TestCombined:
    def test_both_violations_reports_haram_first(self):
        """Haram check takes priority over kill check."""
        passed, reason = HardConstraintGuard.precheck({
            "name": "Cold Pork Salad",
            "name_cn": "冷猪肉沙拉",
            "description": "Chilled pork salad",
            "description_cn": "冷猪肉沙拉",
        })
        assert not passed
        # Haram is checked first, so reason should mention pork/haram
        assert "pork" in reason.lower() or "清" in reason

    def test_clean_proposal_passes_all(self):
        passed, reason = HardConstraintGuard.precheck({
            "name": "Chicken Teriyaki Rice Bowl",
            "name_cn": "照烧鸡肉饭",
            "description": "Grilled chicken with teriyaki sauce on steamed rice",
            "description_cn": "烤鸡肉配照烧酱和米饭",
        })
        assert passed
        assert reason == ""


def test_sake_triggers_veto():
        passed, reason = HardConstraintGuard.check_haram(
            "Sake Steamed Chicken", "", "Chicken steamed with sake", ""
        )
        assert not passed
        assert "sake" in reason


def test_mirin_triggers_veto():
        passed, reason = HardConstraintGuard.check_haram(
            "Mirin Glazed Salmon", "", "Salmon glazed with mirin", ""
        )
        assert not passed
        assert "mirin" in reason


def test_wine_triggers_veto():
        passed, reason = HardConstraintGuard.check_haram(
            "Red Wine Beef Stew", "", "Beef stewed in red wine", ""
        )
        assert not passed
        assert "wine" in reason


def test_gelatin_triggers_veto():
        passed, reason = HardConstraintGuard.check_haram(
            "Gelatin Dessert Bowl", "", "Sweet gelatin dessert", ""
        )
        assert not passed
        assert "gelatin" in reason
