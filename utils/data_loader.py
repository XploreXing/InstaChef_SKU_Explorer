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
        """Call LLM in JSON mode to tag cuisine_type, is_halal_suspect, is_fried."""
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

For each commodity, classify 3 tags based on its name and description:

1. cuisine_type: Choose ONE from [中式, 日式, 韩式, 泰式, 新马, 墨西哥, 其他]
   - 中式: Chinese dishes (fried rice, noodles, braised, mapo, kung pao, dim sum, etc.)
   - 日式: Japanese dishes (teriyaki, udon, ramen, soba, gyudon, sushi, miso, etc.)
   - 韩式: Korean dishes (kimchi, bulgogi, bibimbap, tteokbokki, jjigae, etc.)
   - 泰式: Thai dishes (tom yum, green/red curry, pad thai, basil, etc.)
   - 新马: Singaporean/Malaysian (laksa, nasi lemak, rendang, sambal, mee goreng, etc.)
   - 墨西哥: Mexican dishes (burrito, taco, chipotle, fajita, etc.)
   - 其他: Does not clearly fit any above (Italian pasta, Western, Indian, fusion, salads, desserts, sandwiches, etc.)

2. is_halal_suspect: Mark true ONLY if name/description contains EXPLICIT pork/lard/alcohol references.

   In Singapore context:
   - "Hainanese" alone is NOT suspect — many Hainanese dishes use chicken (e.g. Hainanese chicken rice)
   - "Yam Cake", "Carrot Cake", "Turnip Cake" are vegetable-based — NOT suspect
   - "Bacon" without "turkey" or "chicken" qualifier IS suspect
   - "Ham" without "chicken" or "turkey" qualifier IS suspect

   Explicit pork indicators: "pork", "豚"(buta), "猪", "lard", "char siew" (叉烧= pork), "sausage" without chicken/turkey qualifier, "bangers" (British pork sausages), "Spam", "bacon" (unless chicken/turkey), "ham" (unless chicken/turkey), "prosciutto", "pancetta"

   Alcohol indicators: "mirin", "sake", "cooking wine", "啤酒(beer)", "红酒(red wine)", "米酒". "料酒" when used as cooking wine IS suspect.

   false otherwise.

3. is_fried: CRITICAL — you MUST distinguish deep-fried (炸) from stir-fried/wok-fried (炒).

   In Singapore hawker and local food culture, dishes with "Fried" in the name are almost always STIR-FRIED (wok-fried), NOT deep-fried. Examples:
   - "Fried Carrot Cake" = 炒萝卜糕, wok-fried radish cake — NOT deep-fried
   - "Fried Hokkien Prawn Noodles" = 炒福建虾面, wok-fried — NOT deep-fried
   - "Fried Rice" (炒饭), "Fried Noodles" (炒面), "Fried Udon" (炒乌冬) — all wok-fried
   - "Char Kway Teow" = 炒粿条, wok-fried flat noodles — NOT deep-fried

   Only mark is_fried=true for genuinely DEEP-FRIED items where food is submerged in hot oil:
   - Katsu, tempura, karaage, tonkatsu (deep-fried cutlets)
   - Fried chicken wings, chicken chop (deep-fried), ayam penyet, fish & chips
   - Ebi fry, croquette, korokke
   - Items explicitly described as "deep-fried", "crispy fried", "breaded and fried", "battered"

   If the cooking method is unclear from name alone, default to false.

Return ONLY valid JSON:
{"commodities": [{"id": 94, "cuisine_type": "韩式", "is_halal_suspect": false, "is_fried": false}, ...]}"""

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
            # Fallback: mark all as 其他, not suspect, not fried
            tagged_list = [
                {"id": c.id, "cuisine_type": "其他", "is_halal_suspect": False, "is_fried": False}
                for c in batch
            ]

        # Build lookup by id
        tag_lookup = {t["id"]: t for t in tagged_list}

        results = []
        for c in batch:
            t = tag_lookup.get(c.id, {"cuisine_type": "其他", "is_halal_suspect": False, "is_fried": False})
            results.append(ProcessedCommodity(
                id=c.id,
                name=c.name,
                description=c.description,
                cuisine_type=t["cuisine_type"],
                is_halal_suspect=t["is_halal_suspect"],
                is_fried=t["is_fried"],
            ))
        return results

    def get_by_cuisine(self, cuisine: str) -> list[ProcessedCommodity]:
        return [c for c in self._commodities if c.cuisine_type == cuisine]

    def get_cuisine_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self._commodities:
            counts[c.cuisine_type] = counts.get(c.cuisine_type, 0) + 1
        return counts

    def get_halal_suspects(self) -> list[ProcessedCommodity]:
        return [c for c in self._commodities if c.is_halal_suspect]

    def get_fried_items(self) -> list[ProcessedCommodity]:
        return [c for c in self._commodities if c.is_fried]

    @property
    def all_commodities(self) -> list[ProcessedCommodity]:
        return self._commodities

    @property
    def total_count(self) -> int:
        return len(self._commodities)
