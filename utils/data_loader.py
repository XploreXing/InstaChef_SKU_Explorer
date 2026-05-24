import json


class SKUDataLoader:
    def __init__(self, config: dict):
        self.source = config["data"]["sku_source"]
        self.json_path = config["data"].get("sku_json_path", "")
        self._skus: list[dict] = []

    def load(self) -> list[dict]:
        if self.source == "local_json":
            self._skus = self._load_from_json()
        elif self.source == "postgres":
            raise NotImplementedError("Postgres source not yet implemented")
        elif self.source == "metabase_api":
            raise NotImplementedError("Metabase API source not yet implemented")
        else:
            raise ValueError(f"Unknown sku_source: {self.source}")
        return self._skus

    def _load_from_json(self) -> list[dict]:
        with open(self.json_path, "r") as f:
            return json.load(f)

    def get_by_cuisine(self, cuisine: str) -> list[dict]:
        return [s for s in self._skus if s.get("cuisine") == cuisine]

    def get_cuisine_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self._skus:
            c = s.get("cuisine", "其他")
            counts[c] = counts.get(c, 0) + 1
        return counts

    @property
    def all_skus(self) -> list[dict]:
        return self._skus

    @property
    def total_count(self) -> int:
        return len(self._skus)
