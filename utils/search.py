import os
import re

CUSINE_EN_MAP = {
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Thai": "Thai",
    "Singaporean/Malay": "Singaporean Malay",
    "Mexican": "Mexican",
}

# Content quality: signals that a result is a review/ranking page, not a menu
_REVIEW_TITLE_PATTERNS = [
    r"\bbest\s+\d+\b", r"\btop\s+\d+\b", r"\breview\b", r"\branking\b",
    r"\bwhat\s+is\b", r"\bhow\s+to\b", r"\bguide\b", r"\bvs\b",
]
_REVIEW_URL_PATTERNS = [
    r"/blog/", r"/review", r"/news/", r"/article/", r"/listicle/",
]

# Positive signals: these words suggest actual dish/menu content
_DISH_SIGNAL_WORDS = [
    "menu", "dish", "bowl", "plate", "signature", "grilled", "braised",
    "stir-fried", "roasted", "rice", "noodle", "curry", "sauce",
]


class FoodTrendSearcher:
    def __init__(self, config: dict):
        self.cfg = config["search"]
        self._llm_cfg = config.get("llm")  # ponytail: None in unit tests → regex fallback
        self._client = None

    def _get_client(self):
        if self._client is None:
            from tavily import TavilyClient

            api_key = os.getenv(self.cfg["api_key_env"])
            if not api_key:
                raise ValueError(
                    f"Environment variable {self.cfg['api_key_env']} not set"
                )
            self._client = TavilyClient(api_key=api_key)
        return self._client

    def build_queries(self, cuisine: str) -> list[str]:
        cuisine_en = CUSINE_EN_MAP.get(cuisine, cuisine)
        queries = []
        # 1. General templates
        for template in self.cfg.get("query_templates", []):
            queries.append(template.format(cuisine=cuisine, cuisine_en=cuisine_en))
        # 2. Cuisine-specific overrides
        overrides = self.cfg.get("cuisine_overrides", {}).get(cuisine, {})
        for q in overrides.get("extra_queries", []):
            queries.append(q)
        # 3. Site-targeted queries for menu evidence
        menu_domains = self.cfg.get("menu_domains", [])
        site_templates = self.cfg.get("site_query_templates", [])
        for domain in menu_domains:
            for tpl in site_templates:
                queries.append(
                    tpl.format(domain=domain, cuisine=cuisine, cuisine_en=cuisine_en)
                )
        return queries

    def search_cuisine(self, cuisine: str) -> list[dict]:
        # --- Hop 1: broad cuisine discovery ---
        queries = self.build_queries(cuisine)
        results = []
        for query in queries:
            results.extend(self._do_search(query))
        results = self._deduplicate(results)
        results = self._filter_quality(results, cuisine)
        results = self._tag_source_types(results)

        # --- Hop 2: targeted restaurant-menu deep search ---
        restaurants = self._extract_restaurant_names(results, cuisine)
        if restaurants:
            print(
                f"  [search] Hop 2: deep-searching {len(restaurants)} restaurants "
                f"for {cuisine}: {restaurants[:3]}...",
                flush=True,
            )
            hop2_results = []
            for name in restaurants[:5]:
                for q in self._build_hop2_queries(name, cuisine):
                    hop2_results.extend(self._do_search(q))
            hop2_results = self._deduplicate(hop2_results)
            for r in hop2_results:
                r["source_type"] = "menu"
                r["evidence_level"] = "menu"
                r["hop"] = 2
            results = results + hop2_results

        return results

    def _do_search(self, query: str) -> list[dict]:
        client = self._get_client()
        try:
            response = client.search(
                query=query,
                max_results=self.cfg.get("max_results_per_query", 5),
                include_domains=self.cfg.get("target_domains"),
                time_range=self.cfg.get("time_range"),
            )
            return response.get("results", [])
        except Exception as e:
            print(f"Search failed for '{query}': {e}")
            return []

    def _score_result(self, r: dict, cuisine: str) -> int:
        """Score a search result for menu/dish relevance. 0-100.
        Negative signals (review/blog) penalize; positive signals (dish words) boost."""
        title = (r.get("title", "") or "").lower()
        content = (r.get("content", "") or "").lower()
        url = (r.get("url", "") or "").lower()
        combined = f"{title} {content}"
        score = 50  # neutral start

        # Penalize: review/ranking/blog signals
        for pat in _REVIEW_TITLE_PATTERNS:
            if re.search(pat, title):
                score -= 25
                break
        for pat in _REVIEW_URL_PATTERNS:
            if re.search(pat, url):
                score -= 20
                break

        # Penalize: content too short (likely ad or dead link)
        if len(content) < 100:
            score -= 20

        # Boost: dish/menu signal words
        dish_hits = sum(1 for w in _DISH_SIGNAL_WORDS if w in combined)
        score += min(dish_hits * 5, 25)

        # Boost: cuisine name or cuisine-specific keywords in content
        cuisine_lower = cuisine.lower()
        if cuisine_lower in combined:
            score += 10

        return max(0, min(score, 100))

    def _filter_quality(self, results: list[dict], cuisine: str) -> list[dict]:
        """Keep only results with quality score >= 30, sorted by score desc."""
        scored = [(self._score_result(r, cuisine), r) for r in results]
        kept = [(s, r) for s, r in scored if s >= 30]
        kept.sort(key=lambda x: -x[0])
        filtered = [r for _, r in kept]
        if len(filtered) < len(results):
            print(
                f"  [search] quality filter: {len(filtered)}/{len(results)} "
                f"results kept for {cuisine}",
                flush=True,
            )
        return filtered

    def _classify_source(self, url: str, title: str) -> str:
        """Classify a search result as 'menu' or 'trend' based on URL/domain signals.
        - menu: official restaurant menu pages, /menu paths, brand domains
        - trend: blogs, reviews, social media, ranking pages, everything else
        """
        url_lower = (url or "").lower()
        title_lower = (title or "").lower()

        # Menu signals
        menu_domains = [d.lower() for d in self.cfg.get("menu_domains", [])]
        for domain in menu_domains:
            if domain in url_lower:
                return "menu"
        if "/menu" in url_lower or "/food" in url_lower:
            return "menu"

        # Trend signals
        trend_domain_hints = [
            "reddit", "youtube", "instagram", "tiktok", "facebook",
            "tripadvisor", "eatbook", "misstamchiak", "sethlui",
            "danielfooddiary", "thehoneycombers", "blog", "review",
        ]
        for hint in trend_domain_hints:
            if hint in url_lower:
                return "trend"

        # Default: check title for review/ranking signals
        review_patterns = [r"top\s+\d+", r"best\s+\d+", r"review", r"ranking"]
        for pat in review_patterns:
            if re.search(pat, title_lower):
                return "trend"

        return "trend"  # default fallback

    def _tag_source_types(self, results: list[dict]) -> list[dict]:
        """Add source_type and evidence_level to each result."""
        for r in results:
            source_type = self._classify_source(
                r.get("url", ""), r.get("title", "")
            )
            r["source_type"] = source_type
            r["evidence_level"] = "menu" if source_type == "menu" else "trend"
        return results

    # Dish/category words that leak through the regex fallback as false "restaurant" names
    _DISH_BLACKLIST = {
        # Korean
        "bibimbap", "tteokbokki", "bulgogi", "jjigae", "samgyetang", "kimchi",
        "galbi", "japchae", "pajeon", "samgyeopsal", "doenjang", "tteok",
        # Japanese
        "ramen", "sushi", "sashimi", "udon", "tempura", "tonkotsu", "donburi",
        # Chinese
        "dim sum", "xiao long bao", "wonton", "char siew", "hor fun",
        # Thai
        "pad thai", "tom yum", "green curry", "som tum",
        # Mexican
        "tacos", "burritos", "quesadilla", "nachos", "fajita", "enchilada",
        # Singaporean/Malay
        "laksa", "nasi lemak", "satay", "rendang", "char kway teow",
        "carrot cake", "chicken rice",
    }

    def _extract_restaurant_names(self, results: list[dict], cuisine: str) -> list[str]:
        """Extract candidate restaurant names from Hop 1 results.
        Primary: LLM NER (accurate on truncated listicle snippets, handles
        transliterated/all-caps names the regex misses). Fallback: regex."""
        if self._llm_cfg:
            names = self._llm_extract_restaurant_names(results, cuisine)
            if names:
                return names[:5]
        return self._regex_extract_restaurant_names(results, cuisine)

    def _llm_extract_restaurant_names(self, results: list[dict], cuisine: str) -> list[str]:
        """LLM-based restaurant name extraction. Returns [] on any failure."""
        import json as _json
        import os as _os
        from openai import OpenAI

        # ponytail: cap tokens — top 15 results, 250 chars each
        snippets = []
        for r in results[:15]:
            t = (r.get("title", "") or "").strip()
            c = (r.get("content", "") or "").strip()[:250]
            if not t and not c:
                continue
            snippets.append(f"T: {t}\nC: {c}")
        if not snippets:
            return []

        system = (
            f"Extract actual restaurant/eatery BRAND names from web search snippets about {cuisine} "
            f"food in Singapore. Return ONLY specific dining establishments (restaurants, stalls, "
            f"chains, eateries). EXCLUDE: dish names (bibimbap, tteokbokki, bulgogi, tacos), cuisine "
            f"categories, generic words, listicle titles ('Best/Top N ...'), social media handles, "
            f"district/building names. INCLUDE multi-word or all-caps transliterated names "
            f"(e.g. 'Wang Dae Bak', 'GAHE', 'Go K BBQ'). "
            f'Output JSON: {{"restaurants": ["name", ...]}}. Max 15. Empty list if none.'
        )
        try:
            client = OpenAI(
                base_url=self._llm_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
                api_key=self._llm_cfg.get("api_key")
                or _os.getenv(self._llm_cfg.get("api_key_env", "")),
            )
            resp = client.chat.completions.create(
                model=self._llm_cfg.get("generator_model", "deepseek-ai/DeepSeek-V3"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Snippets:\n" + "\n--\n".join(snippets)},
                ],
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            data = _json.loads(resp.choices[0].message.content)
            names = [str(n).strip() for n in data.get("restaurants", []) if n and str(n).strip()]
            deduped: list[str] = []
            for name in names:
                if not any(
                    name.lower() in e.lower() or e.lower() in name.lower() for e in deduped
                ):
                    deduped.append(name)
            return deduped
        except Exception as e:
            print(
                f"  [search] LLM restaurant extraction failed, falling back to regex: {e}",
                flush=True,
            )
            return []

    def _regex_extract_restaurant_names(self, results: list[dict], cuisine: str) -> list[str]:
        """Regex fallback for restaurant name extraction."""
        candidates: dict[str, int] = {}
        noise_words = {"best", "top", "review", "menu", "food", "restaurant",
                       "restaurants", "singapore", "delivery", "price", "rating",
                       "available", "near", "check", "found", "new", "must", "try",
                       "where", "home", "feature", "guide", "places", "eateries",
                       "cuisine", "dining",
                       "chinese", "japanese", "korean", "thai", "mexican", "malay",
                       cuisine.lower()}

        for r in results:
            title = (r.get("title", "") or "")
            content = (r.get("content", "") or "")
            combined = f"{title} {content}"

            # Pattern: "Name at Location" or "Name, Location – ..."
            for m in re.finditer(
                r"([A-Z][a-z]+(?:['’]s)?(?:\s+(?:&|[A-Z][a-z]+(?:['’]s)?|de|la|el|y|del)){0,4})"
                r"\s+(?:at|in|–|-|—|,)\s+",
                combined,
            ):
                name = m.group(1).strip()
                words_in_name = set(name.lower().split())
                if len(name) > 3 and not words_in_name & noise_words:
                    candidates[name] = candidates.get(name, 0) + 2

            # Pattern: @handle
            for m in re.finditer(r"@(\w{3,})", combined):
                candidates[m.group(1)] = candidates.get(m.group(1), 0) + 2

        # Filter and rank
        ranked = sorted(candidates.items(), key=lambda x: -x[1])
        result = []
        for name, score in ranked:
            if score >= 2 and len(name) >= 4:
                if name.lower() in self._DISH_BLACKLIST:
                    continue
                if not any(name.lower() in e.lower() or e.lower() in name.lower()
                           for e in result):
                    result.append(name)
        return result[:5]

    def _build_hop2_queries(self, restaurant: str, cuisine: str) -> list[str]:
        """Build targeted deep-search queries for a specific restaurant."""
        return [
            f'"{restaurant}" menu Singapore',
            f'"{restaurant}" {cuisine} dishes',
            f'"{restaurant}" signature best seller',
        ]

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        seen_urls = set()
        unique = []
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(r)
        return unique

    def summarize_for_generator(self, results: list[dict], cuisine: str) -> tuple[str, dict[str, str], dict[str, str], dict[str, str]]:
        """Tag search results with [REF_01], [REF_02], etc.
        Returns (summary_text, ref_map, ref_contents, ref_source_types) where:
        - ref_map is {tag: url}
        - ref_contents is {tag: full_content_snippet}
        - ref_source_types is {tag: "menu"|"trend"}"""
        if not results:
            return f"No external trend data found for {cuisine}.", {}, {}, {}

        ref_map: dict[str, str] = {}
        ref_contents: dict[str, str] = {}
        ref_source_types: dict[str, str] = {}
        lines = [f"## External Market Research for {cuisine}\n"]

        for i, r in enumerate(results):
            tag = f"REF_{i + 1:02d}"
            ref_map[tag] = r.get("url", "")
            ref_contents[tag] = r.get("content", "")[:800]  # full for evidence matching
            ref_source_types[tag] = r.get("source_type", "trend")
            summary_content = r.get("content", "")[:300]  # truncated for Generator prompt
            lines.append(
                f"- **[{tag}] {r.get('title', 'Untitled')}** "
                f"({r.get('url', '')}) [{ref_source_types[tag]}]: {summary_content}"
            )

        lines.append(f"\n*{len(results)} sources found. References: {', '.join(ref_map.keys())}.*")
        return "\n".join(lines), ref_map, ref_contents, ref_source_types
