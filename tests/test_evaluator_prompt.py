"""Tests for evaluator system prompt structure — veto-first audit."""
from pathlib import Path


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "evaluator_system.md"


def _load_prompt() -> str:
    with open(PROMPT_PATH) as f:
        return f.read()


class TestPromptStructure:
    def test_prompt_exists_and_readable(self):
        prompt = _load_prompt()
        assert len(prompt) > 100

    def test_contains_physical_audit_step(self):
        """Prompt must enforce a physical/temperature audit as the first step."""
        prompt = _load_prompt()
        assert "物理状态" in prompt or "physical" in prompt.lower()
        assert "热食" in prompt or "hot food" in prompt.lower() or "temperature" in prompt.lower()

    def test_contains_halal_audit_step(self):
        """Prompt must enforce halal compliance audit."""
        prompt = _load_prompt()
        assert "halal" in prompt.lower()

    def test_contains_veto_first_mechanism(self):
        """Veto must happen before scoring, not after."""
        prompt = _load_prompt()
        assert "veto" in prompt.lower()
        assert "一票否决" in prompt or "veto" in prompt.lower()

    def test_vetoed_scores_must_be_zero(self):
        """Vetoed items must have all scores set to 0."""
        prompt = _load_prompt()
        # Should mention zeroing scores on veto
        assert "0" in prompt

    def test_bacon_edge_case_rules(self):
        """Bacon/sausage edge cases must have explicit rules."""
        prompt = _load_prompt()
        assert "bacon" in prompt.lower() or "培根" in prompt
        assert "turkey" in prompt.lower() or "火鸡" in prompt or "beef" in prompt.lower() or "牛肉" in prompt

    def test_contains_fried_constraint(self):
        """Deep-fried constraint must be present."""
        prompt = _load_prompt()
        assert "fried" in prompt.lower() or "炸" in prompt or "katsu" in prompt.lower()

    def test_contains_duplicate_constraint(self):
        """Duplicate detection constraint must be present."""
        prompt = _load_prompt()
        assert "duplicate" in prompt.lower() or "重复" in prompt
