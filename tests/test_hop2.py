"""Tests for 2-hop search flow: restaurant extraction + targeted queries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from utils.search import FoodTrendSearcher


class TestExtractRestaurantNames:
    def test_extracts_from_title_with_location(self):
        config = {"search": {"menu_domains": []}}
        searcher = FoodTrendSearcher(config)
        results = [
            {"title": "Chimichanga at PLQ Mall Singapore: Mexican Food",
             "url": "https://example.com/1", "content": "Tacos and burritos"},
            {"title": "Piedra Negra, Club Street - Mexican Bar",
             "url": "https://example.com/2", "content": "Al pastor tacos"},
        ]
        names = searcher._extract_restaurant_names(results, "Mexican")
        assert len(names) >= 1
        assert any("Chimichanga" in n or "Piedra" in n for n in names)

    def test_extracts_from_at_handle(self):
        config = {"search": {"menu_domains": []}}
        searcher = FoodTrendSearcher(config)
        results = [
            {"title": "Bored Tacos Recently checked out @boredtacossg",
             "url": "https://example.com/1", "content": "A must try Mexican spot"},
        ]
        names = searcher._extract_restaurant_names(results, "Mexican")
        assert len(names) >= 1

    def test_filters_review_titles(self):
        """Titles like 'Top 10 Best Mexican' should NOT be extracted as restaurant names."""
        config = {"search": {"menu_domains": []}}
        searcher = FoodTrendSearcher(config)
        results = [
            {"title": "Top 10 Best Mexican Restaurants in Singapore 2026",
             "url": "https://example.com/1", "content": "Ranking of best spots"},
        ]
        names = searcher._extract_restaurant_names(results, "Mexican")
        assert len(names) == 0

    def test_deduplicates_similar_names(self):
        config = {"search": {"menu_domains": []}}
        searcher = FoodTrendSearcher(config)
        results = [
            {"title": "Papi's Tacos - Mexican Food at Bugis",
             "url": "https://x.com/1", "content": "Papi's Tacos serves great burritos and bowls"},
            {"title": "Papi's Tacos Menu Review",
             "url": "https://x.com/2", "content": "Papi's Tacos menu items"},
        ]
        names = searcher._extract_restaurant_names(results, "Mexican")
        # Should deduplicate to 1 or at most 2
        assert 1 <= len(names) <= 2


class TestHop2Queries:
    def test_builds_targeted_queries(self):
        config = {"search": {"menu_domains": []}}
        searcher = FoodTrendSearcher(config)
        queries = searcher._build_hop2_queries("Chimichanga", "Mexican")
        assert len(queries) == 3
        assert any("menu" in q.lower() for q in queries)
        assert any("mexican" in q.lower() for q in queries)
        assert all("Chimichanga" in q for q in queries)

    def test_queries_include_signature(self):
        config = {"search": {"menu_domains": []}}
        searcher = FoodTrendSearcher(config)
        queries = searcher._build_hop2_queries("Piedra Negra", "Mexican")
        assert any("signature" in q.lower() for q in queries)
