"""Tools available to the Generator Agent."""
import json
import os


# ── Tool implementations ──
_tool_context:dict={}

def validate_halal(ingredients: list[str]) -> str:
    """Check if ingredients comply with halal dietary requirements.
    Deterministic keyword match — no LLM needed."""
    NON_HALAL = {
        "pork", "lard", "bacon", "ham", "alcohol",
        "wine", "sake", "mirin", "gelatin", "猪油",
        "腊肉", "培根", "火腿", "酒",
    }
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
def set_tool_context(existing_skus:list=None):
    global _tool_context
    _tool_context={"existing_skus":existing_skus or []}

def check_duplicate_for_generator(
    name: str,
    name_cn: str,
) -> str:
    """Check if a proposed dish name duplicates any existing SKU.
    Uses the same duplicate_checker module as the orchestrator."""
    from utils.duplication_checker import check_duplicates as _check_dup
   

    # Load existing SKUs for this cuisine (via SKUDataLoader singleton or passed context)
    # This depends on how you wire it — simplest is to make EXISTING_SKUS a module-level
    # or pass it during tool registration
    existing_skus = _tool_context.get("existing_skus", [])
    if not existing_skus:
        return "⚠️ No existing SKU data available for duplicate check."

    is_dup, reason, score = _check_dup(name, name_cn, existing_skus)
    if is_dup:
        return f"❌ DUPLICATE: {reason}"
    return f"✅ No duplicate found. Closest match similarity: {score:.0%}"


# ── OpenAI tool schemas ──

GENERATOR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "validate_halal",
            "description": "Check if a list of ingredients complies with halal dietary requirements. Call this for EVERY proposal before finalizing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ingredient names to check"
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
            "description": "Check if a proposed dish name duplicates any existing SKU in the same cuisine. Call this BEFORE finalizing each proposal.",
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

# ── Tool handler dispatch ──

GENERATOR_HANDLERS = {
    "validate_halal": lambda args: validate_halal(**args),
    "check_duplicate": lambda args: check_duplicate_for_generator(**args),
}