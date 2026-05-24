import csv
import json
import os
from openai import OpenAI
from models import RawCommodity, ProcessedCommodity


class SKUDataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.source = config["data"]["sku_source"]
        self.csv_path = config["data"].get("sku_csv_path", "")
        self.cache_path = config["data"].get(
            "enriched_cache_path", "data/enriched_commodities.json"
        )
        self._commodities: list[ProcessedCommodity] = []
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            cfg = self.config["llm"]
            self._llm_client = OpenAI(
                base_url=cfg["base_url"],
                api_key=os.getenv(cfg["api_key_env"]),
            )
        return self._llm_client

    def load(self) -> list[ProcessedCommodity]:
        if self.source == "local_csv":
            raw = self._load_csv()
            self._commodities = self._enrich(raw)
        elif self.source == "local_json":
            raise NotImplementedError(
                "local_json source deprecated. Use local_csv with current_commodities.csv"
            )
        elif self.source == "postgres":
            raise NotImplementedError("Postgres source not yet implemented")
        else:
            raise ValueError(f"Unknown sku_source: {self.source}")
        return self._commodities

    def _load_csv(self) -> list[RawCommodity]:
        commodities = []
        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                commodities.append(RawCommodity(
                    id=int(row["id"]),
                    name=row["name"].strip(),
                    description=row["description"].strip(),
                ))
        return commodities

    def _load_cache(self) -> dict[int, ProcessedCommodity]:
        """Load cached enrichment results. Returns {id: ProcessedCommodity}."""
        if not os.path.exists(self.cache_path):
            return {}
        with open(self.cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache = {}
        for item in data:
            c = ProcessedCommodity(
                id=item["id"],
                name=item["name"],
                description=item.get("description", ""),
                cuisine_type=item["cuisine_type"],
            )
            cache[c.id] = c
        return cache

    def _save_cache(self, commodities: list[ProcessedCommodity]):
        """Save enrichment results to cache JSON."""
        data = [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "cuisine_type": c.cuisine_type,
            }
            for c in commodities
        ]
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _enrich(self, raw: list[RawCommodity]) -> list[ProcessedCommodity]:
        """Enrich with LLM tags, using cache for incremental updates."""
        cache = self._load_cache()

        if not cache:
            # No cache: full enrichment
            print(
                f"No cache found. "
                f"Running full LLM enrichment on {len(raw)} items..."
            )
            result = self._enrich_with_llm(raw)
            self._save_cache(result)
            return result

        # Cache exists: find delta
        csv_ids = {c.id for c in raw}
        cache_ids = set(cache.keys())
        new_ids = csv_ids - cache_ids
        deleted_ids = cache_ids - csv_ids

        print(
            f"Cache hit. {len(cache)} items cached. "
            f"New: {len(new_ids)}, Deleted: {len(deleted_ids)}"
        )

        if new_ids:
            new_raw = [c for c in raw if c.id in new_ids]
            new_enriched = self._enrich_with_llm(new_raw)
            for c in new_enriched:
                cache[c.id] = c

        for did in deleted_ids:
            del cache[did]

        # Rebuild in CSV order
        result = [cache[c.id] for c in raw if c.id in cache]

        if new_ids or deleted_ids:
            self._save_cache(list(cache.values()))

        return result

    def _enrich_with_llm(
        self, raw_commodities: list[RawCommodity]
    ) -> list[ProcessedCommodity]:
        """Call LLM in JSON mode to tag cuisine_type."""
        batch_size = 30
        results = []

        for i in range(0, len(raw_commodities), batch_size):
            batch = raw_commodities[i:i + batch_size]
            tagged = self._tag_batch(batch)
            results.extend(tagged)

        return results

    def _tag_batch(
        self, batch: list[RawCommodity]
    ) -> list[ProcessedCommodity]:
        items = [
            {"id": c.id, "name": c.name, "description": c.description}
            for c in batch
        ]

        system_prompt = """You are a food data classifier for InstaChef, a halal-compliant smart hot-food vending machine in Singapore.

For each commodity, classify its cuisine_type based on its name and description.

Choose ONE from [中式, 日式, 韩式, 泰式, 新马, 墨西哥, 其他]:

- 中式: Chinese dishes (fried rice, noodles, braised, mapo, kung pao, dim sum, claypot, Hokkien mee, carrot cake, chee cheong fun, etc.)
- 日式: Japanese dishes (teriyaki, udon, ramen, soba, gyudon, sushi, miso, oyakodon, etc.)
- 韩式: Korean dishes (kimchi, bulgogi, bibimbap, tteokbokki, jjigae, Korean army stew, etc.)
- 泰式: Thai dishes (tom yum, green/red curry, pad thai, Thai basil, massaman, etc.)
- 新马: Singaporean/Malaysian/Indonesian (laksa, nasi lemak, rendang, sambal, mee goreng, mee rebus, otah, briyani, etc.)
- 墨西哥: Mexican dishes (burrito, taco, chipotle, fajita, etc.)
- 其他: Italian pasta, Western, Indian, fusion, salads, desserts, sandwiches, etc.

Return ONLY valid JSON:
{"commodities": [{"id": 94, "cuisine_type": "韩式"}, {"id": 218, "cuisine_type": "其他"}]}"""

        user_message = json.dumps(items, ensure_ascii=False)

        try:
            client = self._get_llm_client()
            cfg = self.config["llm"]
            response = client.chat.completions.create(
                model=cfg.get("evaluator_model", cfg.get("generator_model")),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            tagged_list = data.get("commodities", [])
        except Exception as e:
            print(f"LLM tagging failed for batch: {e}")
            # Fallback: mark all as 其他
            tagged_list = [
                {"id": c.id, "cuisine_type": "其他"}
                for c in batch
            ]

        # Build lookup by id
        tag_lookup = {t["id"]: t for t in tagged_list}

        results = []
        for c in batch:
            t = tag_lookup.get(c.id, {"cuisine_type": "其他"})
            results.append(ProcessedCommodity(
                id=c.id,
                name=c.name,
                description=c.description,
                cuisine_type=t["cuisine_type"],
            ))
        return results

    def get_by_cuisine(self, cuisine: str) -> list[ProcessedCommodity]:
        return [c for c in self._commodities if c.cuisine_type == cuisine]

    def get_cuisine_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self._commodities:
            counts[c.cuisine_type] = counts.get(c.cuisine_type, 0) + 1
        return counts

    @property
    def all_commodities(self) -> list[ProcessedCommodity]:
        return self._commodities

    @property
    def total_count(self) -> int:
        return len(self._commodities)
