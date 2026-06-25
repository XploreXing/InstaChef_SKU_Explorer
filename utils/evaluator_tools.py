
"""Tools available to the Evaluator Agent.

Key design:
- search_web_for_eval: verify a dish's existence via web search.
  Returns formatted text for the Evaluator LLM to judge — NOT a bool.
  The LLM reads the search snippets and decides if the dish is real.
"""

def search_web_for_eval(menu: str, cuisine: str) -> str:
    """Search the web to verify a dish's existence.
    Returns formatted search results for the Evaluator LLM to judge."""

    from utils.tavily_client import tavily_search

    lines: list[str] = []
    ref_counter = 0
    queries = [
        f'" Is {menu} a popular {cuisine} dish in Singapore?' ,
        f"if {menu} is a real {cuisine} name?",
    ]
    for q in queries:
        results = tavily_search(q, search_depth="advanced", max_results=3)
        if not results and not lines:
            # Only return "no results" if BOTH queries fail
            continue
        for r in results:
            ref_counter += 1
            lines.append(
                f"[REF_{ref_counter}] {r.get('title', '')}\n"
                f"    URL: {r.get('url', '')}\n"
                f"    {r.get('content', '')[:300]}"
            )

    if not lines:
        return f"⚠️ No search results found for '{menu}' ({cuisine}). This dish may not exist or is very obscure."

    return "\n\n".join(lines)


EVALUATOR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_web_for_eval",
            "description": (
                "Search the web to verify whether a dish actually exists. "
                "Use this when a proposal name sounds suspicious, fabricated, "
                "or like a mashup of two unrelated dishes. "
                "Returns formatted search snippets for you to judge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "menu": {
                        "type": "string",
                        "description": "The dish name to verify (e.g. 'Khao Soi Ramen')",
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "The cuisine type for search context (e.g. 'Thai')",
                    },
                },
                "required": ["menu", "cuisine"],
            },
        },
    },
]