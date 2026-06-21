"""Deterministic duplicate detection — no LLM needed.
Compares proposal names against existing SKU names using token overlap."""
import re
from difflib import SequenceMatcher

_STOP_WORDS = {
    "with", "and", "the", "a", "an", "in", "on", "of", "or", "to",
    "rice", "bowl", "noodles", "noodle", "dish", "style", "served",
    "fresh", "hot", "warm", "new", "special", "signature", "premium",
}

# Ingredient words that are too generic to distinguish dishes
_GENERIC_INGREDIENTS = {
    "rice", "chicken", "beef", "egg", "tofu", "vegetable", "onion",
    "garlic", "ginger", "soy", "oil", "salt", "pepper", "sugar",
}

def _tokenize(name:str) -> set[str]:
    '''Extract meaningful distinguishing tokens from a dish name'''
    tokens=re.findall(r'\w+',name.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t)>2 }

def _distinguishing_tokens(name:str) -> set[str]:
    '''
        Get tokens that actually distinguish dishes (exclude generic ingredients)
    '''
    return _tokenize(name)-_GENERIC_INGREDIENTS #set 的加减法

def check_duplicates(name:str, name_cn:str, existing_skus:list, threshold:float=0.6)-> tuple[bool, str, float]:
    '''
    Check if a proposal duplicates any existing SKU in the same cuisine.

    Uses two signals:
    1. Token overlap (Jaccard): do they share key distinguishing words?
    2. Sequence similarity: are the full names textually similar?

    Returns (is_duplicate: bool, reason: str, similarity: float).
    '''
    new_distinguishing=_distinguishing_tokens(name) | _distinguishing_tokens(name_cn)
    new_all= _tokenize(name) | _tokenize(name_cn)

    best_match_name=None
    best_score=0.0
    for sku in existing_skus:
        sku_name= getattr(sku,'name', '') if hasattr(sku,'name') else str(sku)
        sku_all=_tokenize(sku_name)
        sku_distinguishing=_distinguishing_tokens(sku_name)

        if not new_all or not sku_all:
            continue

        all_intersection = new_all & sku_all
        all_union = new_all | sku_all
        all_jaccard = len(all_intersection) / len(all_union) if all_union else 0

        if new_distinguishing and sku_distinguishing:
            dist_intersection = new_distinguishing & sku_distinguishing
            dist_union = new_distinguishing | sku_distinguishing
            dist_jaccard = len(dist_intersection) / len(dist_union) if dist_union else 0
        else:
            dist_jaccard = 0
        # Signal 3: Sequence similarity on full names
        seq_score = SequenceMatcher(
            None, name.lower(), sku_name.lower()
        ).ratio()

        # Weighted combination: distinguishing token overlap is the strongest signal
        combined = dist_jaccard * 0.5 + all_jaccard * 0.25 + seq_score * 0.25

        if combined > best_score:
            best_score = combined
            best_match_name = sku_name

    if best_score >= threshold and best_match_name:
        return (
            True,
            f"与已有SKU重复: '{best_match_name}' (相似度 {best_score:.0%})",
            best_score,
        )

    return False, "", best_score
