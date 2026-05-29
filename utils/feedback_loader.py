"""HITL feedback loader — standalone module to avoid circular imports.
Used by orchestrator.py (via background thread) to load rejection history."""

import json
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
