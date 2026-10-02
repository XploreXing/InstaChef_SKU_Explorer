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

    # Evaluator is initialised but no API calls are made
    # Generator is per-thread, not stored on orchestrator
    orch._init_agents()
    assert orch.evaluator is not None


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
        assert hasattr(s, "id")
        assert hasattr(s, "name")
        assert hasattr(s, "cuisine_type")

    counts = loader.get_cuisine_counts()
    assert isinstance(counts, dict)

    for cuisine in ["Chinese", "Japanese", "Korean", "Thai", "Singaporean/Malay", "Mexican"]:
        cuisine_skus = loader.get_by_cuisine(cuisine)
        assert isinstance(cuisine_skus, list)


def test_search_query_building_integration():
    """Search query builder produces valid queries."""
    from utils.search import FoodTrendSearcher, CUSINE_EN_MAP
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    searcher = FoodTrendSearcher(config)

    for cuisine in ["Chinese", "Japanese", "Korean", "Thai", "Singaporean/Malay", "Mexican"]:
        queries = searcher.build_queries(cuisine)
        assert len(queries) > 0
        for q in queries:
            assert isinstance(q, str)
            assert len(q) > 0


def test_aggregator_full_flow():
    """Aggregator dedup + sort + write works end-to-end."""
    from models import DishProposal, EvaluationResult, FinalOutput, CuisineResult
    from utils.aggregator import Aggregator

    p1 = DishProposal(1, "Dish A", "Chinese", 5.0, "d", "d", "s")
    p2 = DishProposal(2, "Dish B", "Chinese", 6.0, "d", "d", "s")
    p3 = DishProposal(3, "Dish A", "Chinese", 5.0, "d", "d", "s")  # duplicate name

    def make_eval(p, score):
        return EvaluationResult(
            p, False, None, score / 3.5,
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

    cr = CuisineResult("Chinese", 1, sorted_evals, [])
    output = FinalOutput("2026-05-24", {"Chinese": cr}, 30.0)
    filepath = agg.write_output(output)

    assert os.path.exists(filepath)
    with open(filepath) as f:
        data = json.load(f)
    assert "Chinese" in data["cuisines"]
    assert len(data["cuisines"]["Chinese"]["locked"]) == 2

    os.unlink(filepath)
