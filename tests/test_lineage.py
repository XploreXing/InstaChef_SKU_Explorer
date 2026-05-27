"""Tests for data lineage tracking (source references)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from models import DishProposal


class TestSourceRefs:
    def test_dish_proposal_with_source_refs(self):
        p = DishProposal(
            id=1, name="Test Dish", cuisine="Test",
            price_sgd=5.0, description="A test", differentiation="Unique",
            trend_source="Tavily search",
            source_refs=["REF_01", "REF_03"],
        )
        assert p.source_refs == ["REF_01", "REF_03"]

    def test_dish_proposal_default_empty_source_refs(self):
        p = DishProposal(
            id=1, name="Test Dish", cuisine="Test",
            price_sgd=5.0, description="A test", differentiation="Unique",
            trend_source="Tavily search",
        )
        assert p.source_refs == []

    def test_source_refs_serializable(self):
        from dataclasses import asdict
        p = DishProposal(
            id=1, name="Test Dish", cuisine="Test",
            price_sgd=5.0, description="A test", differentiation="Unique",
            trend_source="Tavily search",
            source_refs=["REF_01"],
        )
        d = asdict(p)
        assert d["source_refs"] == ["REF_01"]


class TestRefValidation:
    def test_empty_source_refs_is_hallucination(self):
        """Proposals with empty source_refs fail validation."""
        source_refs = []
        ref_map = {"REF_01": "https://example.com"}
        valid = bool(source_refs) and all(r in ref_map for r in source_refs)
        assert not valid

    def test_fake_ref_is_hallucination(self):
        """Proposals referencing non-existent tags fail validation."""
        source_refs = ["REF_01", "FAKE_99"]
        ref_map = {"REF_01": "https://example.com"}
        valid = bool(source_refs) and all(r in ref_map for r in source_refs)
        assert not valid

    def test_valid_refs_pass(self):
        """Proposals with all refs present in ref_map pass."""
        source_refs = ["REF_01", "REF_03"]
        ref_map = {"REF_01": "https://a.com", "REF_03": "https://b.com"}
        valid = bool(source_refs) and all(r in ref_map for r in source_refs)
        assert valid


class TestSearchRefTagging:
    def test_summarize_returns_ref_map(self):
        """summarize_for_generator should tag results and return a ref_map."""
        from utils.search import FoodTrendSearcher
        import yaml

        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        searcher = FoodTrendSearcher(config)
        summary, ref_map, ref_contents, ref_source_types = searcher.summarize_for_generator(
            [
                {"title": "Test Article", "url": "https://example.com/1", "content": "Interesting food trend."},
                {"title": "Another Article", "url": "https://example.com/2", "content": "Popular dish."},
            ],
            "Test",
        )

        assert "[REF_01]" in summary
        assert "[REF_02]" in summary
        assert ref_map["REF_01"] == "https://example.com/1"
        assert ref_map["REF_02"] == "https://example.com/2"
        assert len(ref_map) == 2
        assert ref_contents["REF_01"] == "Interesting food trend."

    def test_summarize_empty_results(self):
        from utils.search import FoodTrendSearcher
        import yaml

        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        searcher = FoodTrendSearcher(config)
        summary, ref_map, ref_contents, ref_source_types = searcher.summarize_for_generator([], "Test")

        assert isinstance(summary, str)
        assert ref_map == {}
