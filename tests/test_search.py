from utils.search import FoodTrendSearcher, CUSINE_EN_MAP


def test_cuisine_en_map():
    assert CUSINE_EN_MAP["Chinese"] == "Chinese"
    assert CUSINE_EN_MAP["Japanese"] == "Japanese"
    assert CUSINE_EN_MAP["Korean"] == "Korean"
    assert CUSINE_EN_MAP["Thai"] == "Thai"
    assert CUSINE_EN_MAP["Singaporean/Malay"] == "Singaporean Malay"
    assert CUSINE_EN_MAP["Mexican"] == "Mexican"


def test_build_queries():
    config = {
        "search": {
            "provider": "tavily",
            "api_key_env": "TAVILY_API_KEY",
            "max_results_per_query": 5,
            "query_templates": [
                "{cuisine_en} trending dishes Singapore",
            ],
            "cuisine_overrides": {
                "Mexican": {
                    "extra_queries": [
                        "Stuff'd burrito bowl Singapore",
                    ]
                }
            }
        }
    }
    searcher = FoodTrendSearcher(config)
    queries = searcher.build_queries("Mexican")
    assert len(queries) == 2
    assert queries[0] == "Mexican trending dishes Singapore"
    assert queries[1] == "Stuff'd burrito bowl Singapore"


def test_build_queries_no_overrides():
    config = {
        "search": {
            "provider": "tavily",
            "api_key_env": "TAVILY_API_KEY",
            "max_results_per_query": 5,
            "query_templates": [
                "{cuisine_en} food Singapore",
            ],
            "cuisine_overrides": {}
        }
    }
    searcher = FoodTrendSearcher(config)
    queries = searcher.build_queries("Japanese")
    assert len(queries) == 1
    assert queries[0] == "Japanese food Singapore"


def test_summarize_for_generator_empty():
    config = {"search": {"provider": "tavily", "api_key_env": "KEY"}}
    searcher = FoodTrendSearcher(config)
    summary, ref_map, ref_contents = searcher.summarize_for_generator([], "Chinese")
    assert "No external trend data found" in summary
    assert ref_map == {}
    assert ref_contents == {}


def test_summarize_for_generator_with_results():
    config = {"search": {"provider": "tavily", "api_key_env": "KEY"}}
    searcher = FoodTrendSearcher(config)
    results = [
        {"title": "Din Tai Fung Signature Dishes", "url": "https://example.com/1", "content": "Din Tai Fung new menu launched with spicy dishes and signature rice bowls with braised pork."},
        {"title": "Foodpanda Chinese Menu Items", "url": "https://example.com/2", "content": "Sichuan mala stir-fried noodle dishes growing popular in Singapore restaurants."},
    ]
    summary, ref_map, ref_contents = searcher.summarize_for_generator(results, "Chinese")
    assert "Din Tai Fung" in summary
    assert "[REF_01]" in summary
    assert "[REF_02]" in summary
    assert "Sichuan mala" in summary
    assert "2 sources found" in summary
    assert ref_map["REF_01"] == "https://example.com/1"
    assert ref_map["REF_02"] == "https://example.com/2"
    assert "Din Tai Fung" in ref_contents["REF_01"]


def test_quality_scoring_dish_content():
    """Dish-heavy content scores high."""
    config = {"search": {"provider": "tavily", "api_key_env": "KEY"}}
    searcher = FoodTrendSearcher(config)
    score = searcher._score_result({
        "title": "Grilled Chicken Teriyaki Rice Bowl",
        "url": "https://example.com/menu",
        "content": "Signature grilled chicken teriyaki served on a bowl of steamed rice with braised vegetables and curry sauce.",
    }, "Japanese")
    assert score >= 60


def test_quality_scoring_review_title():
    """Review/ranking titles score low."""
    config = {"search": {"provider": "tavily", "api_key_env": "KEY"}}
    searcher = FoodTrendSearcher(config)
    score = searcher._score_result({
        "title": "Top 10 Best Japanese Restaurants 2026",
        "url": "https://example.com/review/top10",
        "content": "We rank the best Japanese restaurants in Singapore based on reviews.",
    }, "Japanese")
    assert score < 40


def test_quality_scoring_short_content():
    """Very short content scores low."""
    config = {"search": {"provider": "tavily", "api_key_env": "KEY"}}
    searcher = FoodTrendSearcher(config)
    score = searcher._score_result({
        "title": "Click here",
        "url": "https://example.com/ad",
        "content": "Buy now!",
    }, "Chinese")
    assert score < 40


def test_quality_filter_removes_low_quality():
    """Filter should remove low-scoring results."""
    config = {"search": {"provider": "tavily", "api_key_env": "KEY"}}
    searcher = FoodTrendSearcher(config)
    results = [
        {"title": "Top 5 Korean BBQ", "url": "https://x.com/blog/review", "content": "short"},
        {"title": "Korean Bulgogi Rice Bowl Menu", "url": "https://x.com/menu", "content": "Signature bulgogi beef rice bowl with grilled vegetables and special sauce on steamed rice."},
    ]
    filtered = searcher._filter_quality(results, "Korean")
    assert len(filtered) == 1
    assert "Bulgogi" in filtered[0]["title"]
