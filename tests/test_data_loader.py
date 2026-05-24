import tempfile
import os
from utils.data_loader import SKUDataLoader


def test_load_csv():
    """Test CSV parsing (without LLM enrichment — will use fallback)."""
    csv_content = "id,name,description\n"
    csv_content += "1,Test Dish A,A test dish description\n"
    csv_content += "2,Test Dish B,Another test dish\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        config = {
            "data": {"sku_source": "local_csv", "sku_csv_path": tmp_path},
            "llm": {
                "base_url": "https://api.siliconflow.cn/v1",
                "api_key_env": "LLM_API_KEY",
                "generator_model": "test-model",
                "evaluator_model": "test-model",
            },
        }
        loader = SKUDataLoader(config)
        # Since LLM_API_KEY is not set, _enrich_with_llm will use fallback
        # First test CSV parsing only:
        raw = loader._load_csv()
        assert len(raw) == 2
        assert raw[0].id == 1
        assert raw[0].name == "Test Dish A"
        assert raw[1].id == 2
    finally:
        os.unlink(tmp_path)


def test_load_csv_with_bom():
    """Test CSV with BOM character."""
    # Write without BOM in content but using utf-8-sig to add BOM to file
    csv_content = "id,name,description\n"
    csv_content += "1,Test Dish,Description\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                     encoding="utf-8-sig") as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        config = {
            "data": {"sku_source": "local_csv", "sku_csv_path": tmp_path},
            "llm": {
                "base_url": "https://api.example.com/v1",
                "api_key_env": "LLM_API_KEY",
                "generator_model": "test",
                "evaluator_model": "test",
            },
        }
        loader = SKUDataLoader(config)
        raw = loader._load_csv()
        assert len(raw) == 1
        assert raw[0].name == "Test Dish"
    finally:
        os.unlink(tmp_path)


def test_get_by_cuisine_with_processed():
    """Test filtering by cuisine_type with ProcessedCommodity."""
    from models import ProcessedCommodity

    loader = SKUDataLoader.__new__(SKUDataLoader)
    loader._commodities = [
        ProcessedCommodity(1, "A", "", "中式"),
        ProcessedCommodity(2, "B", "", "中式"),
        ProcessedCommodity(3, "C", "", "日式"),
        ProcessedCommodity(4, "D", "", "韩式"),
    ]

    chinese = loader.get_by_cuisine("中式")
    assert len(chinese) == 2

    japanese = loader.get_by_cuisine("日式")
    assert len(japanese) == 1

    mexican = loader.get_by_cuisine("墨西哥")
    assert len(mexican) == 0


def test_get_cuisine_counts():
    from models import ProcessedCommodity

    loader = SKUDataLoader.__new__(SKUDataLoader)
    loader._commodities = [
        ProcessedCommodity(1, "A", "", "中式"),
        ProcessedCommodity(2, "B", "", "中式"),
        ProcessedCommodity(3, "C", "", "日式"),
    ]

    counts = loader.get_cuisine_counts()
    assert counts["中式"] == 2
    assert counts["日式"] == 1


def test_unsupported_source_raises():
    config = {
        "data": {"sku_source": "postgres", "sku_csv_path": ""},
        "llm": {},
    }
    loader = SKUDataLoader(config)
    try:
        loader.load()
        assert False, "Should have raised"
    except NotImplementedError:
        pass
