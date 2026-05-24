import json
import tempfile
import os
from utils.data_loader import SKUDataLoader


def test_load_from_json():
    skus = [
        {"id": 1, "name": "Test Dish", "cuisine": "中式"},
        {"id": 2, "name": "Test Dish 2", "cuisine": "日式"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(skus, f)
        tmp_path = f.name

    try:
        config = {"data": {"sku_source": "local_json", "sku_json_path": tmp_path}}
        loader = SKUDataLoader(config)
        result = loader.load()
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["cuisine"] == "日式"
    finally:
        os.unlink(tmp_path)


def test_get_by_cuisine():
    skus = [
        {"id": 1, "name": "A", "cuisine": "中式"},
        {"id": 2, "name": "B", "cuisine": "中式"},
        {"id": 3, "name": "C", "cuisine": "日式"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(skus, f)
        tmp_path = f.name

    try:
        config = {"data": {"sku_source": "local_json", "sku_json_path": tmp_path}}
        loader = SKUDataLoader(config)
        loader.load()
        chinese = loader.get_by_cuisine("中式")
        assert len(chinese) == 2
        japanese = loader.get_by_cuisine("日式")
        assert len(japanese) == 1
        mexican = loader.get_by_cuisine("墨西哥")
        assert len(mexican) == 0
    finally:
        os.unlink(tmp_path)


def test_get_cuisine_counts():
    skus = [
        {"id": 1, "name": "A", "cuisine": "中式"},
        {"id": 2, "name": "B", "cuisine": "中式"},
        {"id": 3, "name": "C", "cuisine": "日式"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(skus, f)
        tmp_path = f.name

    try:
        config = {"data": {"sku_source": "local_json", "sku_json_path": tmp_path}}
        loader = SKUDataLoader(config)
        loader.load()
        counts = loader.get_cuisine_counts()
        assert counts["中式"] == 2
        assert counts["日式"] == 1
    finally:
        os.unlink(tmp_path)


def test_unsupported_source_raises():
    config = {"data": {"sku_source": "postgres", "sku_json_path": ""}}
    loader = SKUDataLoader(config)
    try:
        loader.load()
        assert False, "Should have raised"
    except NotImplementedError:
        pass
