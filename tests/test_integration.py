import json
import tempfile
import os
from models import OrchestratorState


def test_orchestrator_state_transitions(monkeypatch):
    """Verify state machine flows correctly (without actual LLM calls)."""
    # Set dummy API key so OpenAI client construction succeeds
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-for-testing")
    from orchestrator import Orchestrator

    orch = Orchestrator("config.yaml")
    assert orch.state == OrchestratorState.IDLE

    orch.load_skus()
    assert orch.state == OrchestratorState.INIT
    assert len(orch.existing_skus) > 0

    # Generator/evaluator/searcher are initialised but no API calls are made
    orch._init_agents()
    assert orch.generator is not None
    assert orch.evaluator is not None
    assert orch.searcher is not None


def test_data_loader_integration():
    """SKU loader returns valid data with expected structure."""
    from utils.data_loader import SKUDataLoader
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    loader = SKUDataLoader(config)
    skus = loader.load()

    assert isinstance(skus, list)
    assert len(skus) > 0

    for s in skus:
        assert "id" in s
        assert "name" in s
        assert "cuisine" in s

    counts = loader.get_cuisine_counts()
    assert isinstance(counts, dict)

    for cuisine in ["中式", "日式", "韩式", "泰式", "新马", "墨西哥"]:
        cuisine_skus = loader.get_by_cuisine(cuisine)
        assert isinstance(cuisine_skus, list)


def test_search_query_building_integration():
    """Search query builder produces valid queries."""
    from utils.search import FoodTrendSearcher, CUSINE_EN_MAP
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    searcher = FoodTrendSearcher(config)

    for cuisine in ["中式", "日式", "韩式", "泰式", "新马", "墨西哥"]:
        queries = searcher.build_queries(cuisine)
        assert len(queries) > 0
        for q in queries:
            assert isinstance(q, str)
            assert len(q) > 0


def test_aggregator_full_flow():
    """Aggregator dedup + sort + write works end-to-end."""
    from models import DishProposal, EvaluationResult, FinalOutput, CuisineResult
    from utils.aggregator import Aggregator

    p1 = DishProposal(1, "Dish A", "中式", 5.0, "d", "d", "s")
    p2 = DishProposal(2, "Dish B", "中式", 6.0, "d", "d", "s")
    p3 = DishProposal(3, "Dish A", "中式", 5.0, "d", "d", "s")  # duplicate name

    def make_eval(p, score):
        return EvaluationResult(
            p, False, None, score / 4, score / 3.5,
            score / 2.5, score, True, "r"
        )

    evals = [make_eval(p1, 90), make_eval(p2, 85), make_eval(p3, 80)]
    deduped = Aggregator.deduplicate(evals)
    assert len(deduped) == 2

    sorted_evals = Aggregator.sort_by_score(deduped)
    assert sorted_evals[0].total_score >= sorted_evals[1].total_score

    tmpdir = tempfile.mkdtemp()
    config = {
        "output": {
            "base_dir": tmpdir,
            "filename_prefix": "test",
            "format": "json",
        }
    }
    agg = Aggregator(config)

    cr = CuisineResult("中式", 1, sorted_evals, [])
    output = FinalOutput("2026-05-24", {"中式": cr}, 30.0)
    filepath = agg.write_output(output)

    assert os.path.exists(filepath)
    with open(filepath) as f:
        data = json.load(f)
    assert "中式" in data["cuisines"]
    assert len(data["cuisines"]["中式"]["locked"]) == 2

    os.unlink(filepath)
