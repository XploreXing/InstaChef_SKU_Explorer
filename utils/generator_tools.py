
"""Tools available to the Generator Agent.

Key design:
- ToolContext: per-Agent-instance state (thread-safe, no global vars)
- search_web: lightweight single-query search
- discover_cuisine: heavy multi-hop cuisine research (reuses FoodTrendSearcher logic)
- validate_halal / check_duplicate: retained from previous version, now ctx-aware
"""

import json
from models import ToolContext



# ── Tool implementations ────────────────────────────────────────────────

NON_HALAL = {
    "pork", "lard", "bacon", "ham", "alcohol",
    "wine", "sake", "mirin", "gelatin",
    "猪油", "腊肉", "培根", "火腿", "酒",
}


def validate_halal(ingredients: list[str]) -> str:
    """Check if ingredients comply with halal dietary requirements.
    Deterministic keyword match — no LLM needed."""
    issues = []
    for ing in ingredients:
        ing_lower = ing.lower()
        for nh in NON_HALAL:
            if nh in ing_lower:
                issues.append(f"❌ {ing} — contains '{nh}'")
                break

    if issues:
        return "NON-HALAL DETECTED:\n" + "\n".join(issues)
    return "✅ All ingredients appear halal-compliant."


def check_duplicate_for_generator(name: str, name_cn: str, ctx: ToolContext) -> str:
    """Check if a proposed dish name duplicates any existing SKU.
    Uses ctx.existing_skus instead of a global variable."""
    from utils.duplication_checker import check_duplicates as _check_dup

    existing_skus = ctx.existing_skus
    if not existing_skus:
        return "This cuisine is now a blue ocean series in our business system."

    is_dup, reason, score = _check_dup(name, name_cn, existing_skus)
    if is_dup:
        return f"❌ DUPLICATE: {reason}"
    return f"✅ No duplicate found. Closest match similarity: {score:.0%}"


def search_web(query: str, ctx: ToolContext) -> str:
    """Lightweight web search — single Tavily query.

    Results are tagged with REF_XX and appended to ctx.source_urls
    / ctx.source_contents for traceability and source_urls output.
    """
    from utils.tavily_client import tavily_search

    max_results = (
        ctx.config.get("search", {}).get("max_results_per_query", 5)
    )
    results = tavily_search(query, max_results=max_results,search_depth="advanced")
    if not results:
        return "No results found."

    lines = []
    offset = len(ctx.source_urls)
    for i, r in enumerate(results):
        tag = f"REF_{offset + i + 1:02d}"
        url = r.get("url", "")
        ctx.source_urls[tag] = url
        ctx.source_contents[tag] = r.get("content", "")[:800]
        lines.append(
            f"[{tag}] {r.get('title', '')}\n"
            f"    URL: {url}\n"
            f"    {r.get('content', '')[:300]}"
        )
    return "\n\n".join(lines)


def discover_cuisine(cuisine: str, ctx: ToolContext) -> str:
    """Full cuisine-level search with Hop1 + Hop2.

    Reuses FoodTrendSearcher's query building, filtering, and
    classification logic, but calls tavily_search() instead of
    the searcher's own _do_search() so requests go through the
    shared Semaphore.

    Results are tagged and appended to ctx. Call ONCE per cuisine
    (Round 1 only) — later rounds should use search_web for
    follow-up queries.
    """
    from utils.search import FoodTrendSearcher
    from utils.tavily_client import tavily_search

    # Build a config-like dict so FoodTrendSearcher can read search settings
    searcher = FoodTrendSearcher(ctx.config)

    # --- Hop 1: broad cuisine discovery ---
    queries = searcher.build_queries(cuisine)
    all_results = []
    for q in queries:
        all_results.extend(
            tavily_search(
                q,
                max_results=searcher.cfg.get("max_results_per_query", 5),
                search_depth="basic"
            )
        )
    all_results = searcher._deduplicate(all_results)
    all_results = searcher._filter_quality(all_results, cuisine)
    all_results = searcher._tag_source_types(all_results)

    # --- Hop 2: restaurant-menu deep search ---
    restaurants = searcher._extract_restaurant_names(all_results, cuisine)
    if restaurants:
        print(
            f"  [discover_cuisine] Hop 2: {len(restaurants)} restaurants "
            f"for {cuisine}: {restaurants[:3]}...",
            flush=True,
        )
        hop2_results = []
        for name in restaurants[:5]:
            for q in searcher._build_hop2_queries(name, cuisine):
                hop2_results.extend(
                    tavily_search(
                        q,
                        max_results=searcher.cfg.get("max_results_per_query", 5),
                        search_depth="advanced"
                    )
                )
        hop2_results = searcher._deduplicate(hop2_results)
        for r in hop2_results:
            r["source_type"] = "menu"
            r["evidence_level"] = "menu"
            r["hop"] = 2
        all_results = all_results + hop2_results

    # --- Tag results and append to ctx ---
    lines = [f"## External Market Research for {cuisine}\n"]
    offset = len(ctx.source_urls)
    for i, r in enumerate(all_results):
        tag = f"REF_{offset + i + 1:02d}"
        url = r.get("url", "")
        stype = r.get("source_type", "trend")
        ctx.source_urls[tag] = url
        ctx.source_contents[tag] = r.get("content", "")[:800]
        lines.append(
            f"- **[{tag}] {r.get('title', '')}** "
            f"({url}) [{stype}]: {r.get('content', '')[:300]}"
        )

    lines.append(f"\n*{len(all_results)} sources found. References: {', '.join(ctx.source_urls.keys())}.*")
    return "\n".join(lines)


# ── OpenAI tool schemas ─────────────────────────────────────────────────

GENERATOR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "discover_cuisine",
            "description": (
                "Execute a full cuisine-level search (broad discovery + "
                "targeted restaurant menu deep-search). Call this FIRST "
                "when starting a new cuisine. Returns tagged market "
                "research references. Expensive — do NOT call again in "
                "later rounds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cuisine": {
                        "type": "string",
                        "description": "Cuisine name to research, e.g. 'Thai', 'Korean'",
                    },
                },
                "required": ["cuisine"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for a specific query. Lightweight, call "
                "anytime for targeted follow-up research. Returns tagged "
                "references."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_halal",
            "description": (
                "Check if a list of ingredients complies with halal "
                "dietary requirements. Call this for EVERY proposal "
                "before finalizing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ingredient names to check",
                    },
                },
                "required": ["ingredients"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_duplicate",
            "description": (
                "Check if a proposed dish name duplicates any existing "
                "SKU in the same cuisine. Call this BEFORE finalizing "
                "each proposal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "English dish name to check",
                    },
                    "name_cn": {
                        "type": "string",
                        "description": "Chinese dish name to check",
                    },
                },
                "required": ["name", "name_cn"],
            },
        },
    },
]


# ── Tool handler dispatch ───────────────────────────────────────────────

def build_handlers(ctx: ToolContext) -> dict: #这个可以参考是在dispatchtools中绑定工具使用相关的上下文，这一步用处是什么？需要思考
    """Build tool handler dispatch with bound ToolContext.

    Each GeneratorAgent instance calls this with its own ToolContext,
    so handlers are thread-safe and don't rely on global state.
    """
    return {
        "discover_cuisine": lambda args: discover_cuisine(
            cuisine=args["cuisine"], ctx=ctx
        ),
        "search_web": lambda args: search_web(
            query=args["query"], ctx=ctx
        ),
        "validate_halal": lambda args: validate_halal(**args),
        "check_duplicate": lambda args: check_duplicate_for_generator(
            name=args["name"], name_cn=args["name_cn"], ctx=ctx
        ),
    }
