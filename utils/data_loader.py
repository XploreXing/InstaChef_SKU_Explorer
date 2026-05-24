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
            self._commodities = self._enrich_with_llm(raw)
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
