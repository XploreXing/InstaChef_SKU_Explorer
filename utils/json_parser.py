"""Robust JSON parser for LLM responses.

Handles common LLM output issues:
- Empty / whitespace-only content
- Markdown code fences (```json ... ``` or ``` ... ```)
- Extraneous text before/after the JSON blob
- DeepSeek-V3 quirks (sometimes wraps output in fences or adds commentary)
"""

import json
import re


def parse_llm_json(content: str, top_key: str | None = None) -> dict | list | None:
    """Parse JSON from LLM response content with defensive stripping.

    Args:
        content: Raw message.content from the LLM.
        top_key: If set, extract this key from the parsed dict.
                 e.g. top_key="proposals" returns data["proposals"].

    Returns:
        Parsed JSON object, or the value of top_key, or None on failure.
    """
    if not content or not content.strip():
        return None

    stripped = content.strip()

    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.match(
        r"^```(?:json)?\s*\n?(.*?)\n?\s*```$",
        stripped,
        re.DOTALL,
    )
    if fence_match:
        stripped = fence_match.group(1).strip()

    # 2. Direct parse attempt
    try:
        data = json.loads(stripped)
        if top_key is not None and isinstance(data, dict):
            return data.get(top_key)
        return data
    except json.JSONDecodeError:
        pass

    # 3. Last resort: extract first balanced {...} or [...] blob
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(start_char)
        if start != -1:
            end = stripped.rfind(end_char)
            if end > start:
                try:
                    data = json.loads(stripped[start:end + 1])
                    if top_key is not None and isinstance(data, dict):
                        return data.get(top_key)
                    return data
                except json.JSONDecodeError:
                    pass

    return None
