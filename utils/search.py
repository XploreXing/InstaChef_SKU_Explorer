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
        for template in self.cfg.get("query_templates", []):
            queries.append(template.format(cuisine=cuisine, cuisine_en=cuisine_en))
        overrides = self.cfg.get("cuisine_overrides", {}).get(cuisine, {})
        for q in overrides.get("extra_queries", []):
            queries.append(q)
        return queries

    def search_cuisine(self, cuisine: str) -> list[dict]:
        queries = self.build_queries(cuisine)
        results = []
        for query in queries:
            results.extend(self._do_search(query))
        results = self._deduplicate(results)
        results = self._filter_quality(results, cuisine)
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

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        seen_urls = set()
        unique = []
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(r)
        return unique

    def summarize_for_generator(self, results: list[dict], cuisine: str) -> tuple[str, dict[str, str]]:
        """Tag search results with [REF_01], [REF_02], etc. and return a mapping.
        Returns (summary_text, ref_map) where ref_map is {tag: url}."""
        if not results:
            return f"No external trend data found for {cuisine}.", {}

        ref_map: dict[str, str] = {}
        lines = [f"## External Market Research for {cuisine}\n"]

        for i, r in enumerate(results):
            tag = f"REF_{i + 1:02d}"
            ref_map[tag] = r.get("url", "")
            content = r.get("content", "")[:300]
            lines.append(
                f"- **[{tag}] {r.get('title', 'Untitled')}** "
                f"({r.get('url', '')}): {content}"
            )

        lines.append(f"\n*{len(results)} sources found. References: {', '.join(ref_map.keys())}.*")
        return "\n".join(lines), ref_map
