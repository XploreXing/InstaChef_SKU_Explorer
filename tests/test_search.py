from utils.search import FoodTrendSearcher, CUSINE_EN_MAP


def test_cuisine_en_map():
    assert CUSINE_EN_MAP["中式"] == "Chinese"
    assert CUSINE_EN_MAP["日式"] == "Japanese"
    assert CUSINE_EN_MAP["韩式"] == "Korean"
    assert CUSINE_EN_MAP["泰式"] == "Thai"
    assert CUSINE_EN_MAP["新马"] == "Singaporean Malay"
    assert CUSINE_EN_MAP["墨西哥"] == "Mexican"


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
                "墨西哥": {
                    "extra_queries": [
                        "Stuff'd burrito bowl Singapore",
                    ]
                }
            }
        }
    }
    searcher = FoodTrendSearcher(config)
    queries = searcher.build_queries("墨西哥")
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
    queries = searcher.build_queries("日式")
    assert len(queries) == 1
    assert queries[0] == "Japanese food Singapore"


def test_summarize_for_generator_empty():
    config = {"search": {"provider": "tavily", "api_key_env": "KEY"}}
    searcher = FoodTrendSearcher(config)
    summary = searcher.summarize_for_generator([], "中式")
    assert "No external trend data found" in summary


def test_summarize_for_generator_with_results():
    config = {"search": {"provider": "tavily", "api_key_env": "KEY"}}
    searcher = FoodTrendSearcher(config)
    results = [
        {"title": "Best Chinese Food 2026", "url": "https://example.com/1", "content": "Din Tai Fung new menu launched with spicy dishes."},
        {"title": "Foodpanda Trends", "url": "https://example.com/2", "content": "Sichuan mala growing 30% YoY."},
    ]
    summary = searcher.summarize_for_generator(results, "中式")
    assert "Best Chinese Food 2026" in summary
    assert "Din Tai Fung" in summary
    assert "Sichuan mala" in summary
    assert "2 sources found" in summary
