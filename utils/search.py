import os

CUSINE_EN_MAP = {
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Thai": "Thai",
    "Singaporean/Malay": "Singaporean Malay",
    "Mexican": "Mexican",
}


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
        return self._deduplicate(results)

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
