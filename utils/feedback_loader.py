"""HITL feedback loader — standalone module to avoid circular imports.
Used by orchestrator.py (via background thread) to load rejection history
and build blacklists + semantic summaries for Generator guidance."""

import json
import re
from collections import defaultdict
from pathlib import Path


def load_all_rejections(log_dir: Path) -> list[dict]:
    """Load all rejection feedback from a directory."""
    all_rejections: list[dict] = []
    if not log_dir.exists():
        return all_rejections
    for f in sorted(log_dir.glob("rejections_*.json")):
        try:
            data = json.loads(f.read_text())
            all_rejections.extend(data.get("rejections", []))
        except (json.JSONDecodeError, OSError):
            pass
    return all_rejections


def _normalize_name(name: str, sort_tokens: bool = False) -> str:
    """Normalize a dish name for deterministic matching.
    - lowercase
    - remove punctuation
    - normalize whitespace
    - optional: sort tokens for order-independent matching
    """
    # Remove punctuation that should just disappear (no space left)
    cleaned = re.sub(r"['’]", "", name.lower())
    # Replace other punctuation with space
    cleaned = re.sub(r"[&\",.()!?:;]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if sort_tokens:
        tokens = sorted(cleaned.split())
        return " ".join(tokens)
    return cleaned


def build_blacklist(rejections: list[dict], cuisine: str) -> set[str]:
    """Build a deterministic blacklist of normalized dish names
    that have been rejected for the given cuisine.
    Returns a set of order-sorted normalized names for exact matching."""
    blacklist: set[str] = set()
    for r in rejections:
        if r.get("cuisine") == cuisine:
            normalized = _normalize_name(r.get("proposal_name", ""), sort_tokens=True)
            if normalized:
                blacklist.add(normalized)
    return blacklist


def build_feedback_summary(rejections: list[dict], cuisine: str) -> str:
    """Build a short natural-language summary of rejection patterns
    for the given cuisine, to be injected into the Generator prompt."""
    cuisine_rejections = [
        r for r in rejections if r.get("cuisine") == cuisine
    ]
    if not cuisine_rejections:
        return ""

    # Group by reason
    by_reason: dict[str, list[str]] = defaultdict(list)
    for r in cuisine_rejections:
        reason = r.get("reason_label", "其他")
        name = r.get("proposal_name", "")
        by_reason[reason].append(name)

    lines = [
        "\n## HITL Feedback: Human Expert Rejection Patterns",
        "The following dishes were rejected by human reviewers. Learn from these patterns:\n",
    ]

    for reason, names in by_reason.items():
        names_str = "、".join(names[:3])
        if len(names) > 3:
            names_str += f" 等{len(names)}个"
        lines.append(f"- **{reason}**: {names_str}")

    lines.append(
        f"\n*Use this feedback to avoid proposing similar dishes. "
        f"Focus on what worked and what the reviewers preferred.*"
    )
    return "\n".join(lines)
