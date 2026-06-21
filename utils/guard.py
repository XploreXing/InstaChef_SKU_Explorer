"""Hard-coded high-voltage guard rails.
100% deterministic Python pre-checks — no LLM involved.
Runs BEFORE the Evaluator to catch obvious violations instantly."""

import re


# --- Task 1: Absolute prohibition word lists ---

HARAM_WORDS = [
    "猪肉", "pork",
    "酒", "alcohol",
    "猪油", "lard",
    "火腿", "ham",
    "腊肉", "培根",
]
FRIED_WORDS= [
    "katsu", "tempura", "karaage",
    "tonkatsu", "炸鸡", "炸猪排", "炸虾",
    "ayam penyet", "deep fried", "deep-fried",
    "fried chicken wings", "fried chicken wing",
]
KILL_WORDS = [
    "冷面", "凉皮", "冰镇", "冷盘",
    "凉拌菜", "凉拌",
    "沙拉", "salad",
    "冷", "冻",
    "生食", "刺身", "sashimi",
    "冰淇淋", "ice cream",
    # Fruit-based cold preparations
    "芒果碗", "mango bowl",
    "水果碗", "fruit bowl",
    "果昔碗", "smoothie bowl",
    "酸奶碗", "yogurt bowl",
]

HAWKER_STAPLE_WORDS = [
    "chicken rice", "鸡饭", "海南鸡饭",
    "char siew", "char siu", "叉烧",
    "char kway teow", "炒粿条",
    "fishball noodles", "鱼丸面", "鱼圆面",
    "beef hor fun", "牛肉河粉",
    "mee siam", "米暹",
    "wanton mee", "云吞面", "馄饨面",
    "bak chor mee", "肉脞面",
    "laksa", "叻沙",          # 小贩核心品类
    "nasi lemak", "椰浆饭",   # 无处不在
]

# Words that indicate the dish REQUIRES cold serving
COLD_REQUIRED_PATTERNS = [
    r"cold\s+brew",
    r"冰镇",
    r"冷藏",
    r"冷吃",
    r"serve\s+cold",
    r"serve\s+chilled",
]


class HardConstraintGuard:
    """Deterministic pre-flight checks. Vetoes proposals that unambiguously
    violate hard constraints based on string matching alone — no LLM needed."""

    @staticmethod
    def _normalize(text: str) -> str:
        return text.lower().strip()

    @staticmethod
    def _contains_any(text: str, words: list[str]) -> tuple[bool, str]:
        """Check if text contains any of the given words (case-insensitive).
        Returns (found: bool, matched_word: str)."""
        normalized = HardConstraintGuard._normalize(text)
        for word in words:
            if HardConstraintGuard._normalize(word) in normalized:
                return True, word
        return False, ""

    @staticmethod
    def _matches_any_pattern(text: str, patterns: list[str]) -> tuple[bool, str]:
        normalized = HardConstraintGuard._normalize(text)
        for pattern in patterns:
            if re.search(pattern, normalized):
                return True, pattern
        return False, ""

    @classmethod
    def check_haram(cls, name: str, name_cn: str,
                    description: str, description_cn: str) -> tuple[bool, str]:
        """Check for haram (non-halal) ingredients.
        Returns (passed: bool, veto_reason: str)."""
        combined = f"{name} {name_cn} {description} {description_cn}"
        found, word = cls._contains_any(combined, HARAM_WORDS)
        if found:
            return False, f"高压线熔断：检测到清真违规词 '{word}'"
        return True, ""
    @classmethod
    def check_fried(cls,name:str, name_cn:str,description:str,description_cn:str)->tuple[bool,str]:
        """Check for fried ingredients.
           Returns (passed:bool, veto_reson:str)
        """
        combined=f'{name} {name_cn} {description} {description_cn}'
        found,word=cls._contains_any(combined,FRIED_WORDS)
        if found:
            return False, f"高压线熔断：检测到油炸物相关描述"
    @classmethod
    def check_hawker_staple(cls,name:str, name_cn:str,description:str,description_cn:str)->tuple[bool,str]:
        """Check for ingredients sold in hawker centers.
           Returns (passed:bool, veto_reson:str)
        """
        combined=f'{name} {name_cn} {description} {description_cn}'
        found,word=cls._contains_any(combined,HAWKER_STAPLE_WORDS)
        if found:
            return False, f"高压线熔断：检测到小贩中心常卖食物相关描述"
    @classmethod
    def check_kill(cls, name: str, name_cn: str,
                   description: str, description_cn: str) -> tuple[bool, str]:
        """Check for cold/raw dishes that don't fit the vending machine format.
        Returns (passed: bool, veto_reason: str)."""
        combined = f"{name} {name_cn} {description} {description_cn}"

        # First, check for patterns that unequivocally require cold serving
        found, pattern = cls._matches_any_pattern(combined, COLD_REQUIRED_PATTERNS)
        if found:
            return False, f"高压线熔断：检测到冷食/生食相关描述 '{pattern}'"

        # Then check kill words — but with context awareness:
        # "hot" + "cold" adjacent = OK (describing contrast, not serving temp)
        # "serve cold" or standalone "salad" = NOT OK
        found, word = cls._contains_any(combined, KILL_WORDS)
        if found:
            # Allow "salad" if it's a warm salad (e.g. "warm potato salad")
            if cls._normalize(word) in ("沙拉", "salad"):
                if cls._contains_any(combined, ["warm", "hot", "热", "温", "烤", "grilled", "roasted"])[0]:
                    return True, ""
            # Allow "冻" if context is "冻豆腐"(frozen tofu — cooking ingredient)
            if cls._normalize(word) == "冻":
                if cls._contains_any(combined, ["冻豆腐", "冻肉"])[0]:
                    return True, ""
            return False, f"高压线熔断：检测到冷食/生食关键词 '{word}'"

        return True, ""

    @classmethod
    def precheck(cls, proposal: dict) -> tuple[bool, str]:
        """Run all hard-coded checks on a proposal dict.
        Returns (passed: bool, veto_reason: str).
        If passed, proposal can proceed to Evaluator LLM.
        If vetoed, it gets EvaluationResult(vetoed=True, scores=0) immediately."""
        name = proposal.get("name", "")
        name_cn = proposal.get("name_cn", "")
        description = proposal.get("description", "")
        description_cn = proposal.get("description_cn", "")

        # 1. Haram check (highest priority)
        passed, reason = cls.check_haram(name, name_cn, description, description_cn)
        if not passed:
            return False, reason

        # 2. Kill check (physical format mismatch)
        passed, reason = cls.check_kill(name, name_cn, description, description_cn)

        # 3. Fried check
        passed,reason=cls.check_fried(name,name_cn, description, description_cn)

        # 4. Hawker Staple check
        passed,reason=cls.check_hawker_staple(name,name_cn, description, description_cn)
        if not passed:
            return False, reason

        return True, ""


def get_guard_growth_suggestions(rejections: list[dict]) -> dict:
    """Analyze HITL feedback and propose guard rule updates.
    Returns {rule_type: [suggested_words_to_add]}.
    Only triggers when same reason appears >= 3 times."""
    from collections import Counter

    reasons = Counter(r.get("reason_code", "") for r in rejections)
    suggestions: dict[str, list[str]] = {}

    for reason, count in reasons.items():
        if count < 3:
            continue

        if reason == "cold_food":
            words = set()
            for r in rejections:
                if r.get("reason_code") == "cold_food":
                    for w in r.get("proposal_name", "").lower().split():
                        w_clean = w.strip(",.()")
                        if len(w_clean) > 3:
                            words.add(w_clean)
            if words:
                suggestions["kill_words"] = sorted(words)

        elif reason == "non_halal":
            words = set()
            for r in rejections:
                if r.get("reason_code") == "non_halal":
                    for w in r.get("proposal_name", "").lower().split():
                        w_clean = w.strip(",.()")
                        if len(w_clean) > 3:
                            words.add(w_clean)
            if words:
                suggestions["haram_words"] = sorted(words)

    return suggestions
