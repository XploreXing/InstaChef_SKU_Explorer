"""Lightweight token estimation for observability metrics.

Used only as a fallback when a provider doesn't return usage info. Prefer the
real `usage` from ModelResponse (accurate, free); this is a rough estimate
for cost/latency tracing when usage is absent.

COEFFICIENT
1 Chinese char ≈ 1-2 tokens; 1 English word ≈ 1.3 tokens (≈4 chars/token).
For mixed zh/en content (SKU descriptions, prompts), len * 0.6 is a
reasonable compromise (zh leans ~1.0, en leans ~0.25). This is intentionally
NOT tiktoken — tiktoken (cl100k_base) targets GPT BPE and is inaccurate for
DeepSeek/GLM/Qwen anyway, plus it adds a dependency + startup cost that
isn't justified for observability-grade estimates.
"""

from __future__ import annotations


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate total token count from a message list.

    Counts content length across all messages and applies the 0.6 coefficient.
    Handles non-string content (tool calls etc.) by stringifying.
    """
    total_chars = 0
    for m in messages:
        content = m.get("content", "")
        if not isinstance(content, str):
            # Some messages (tool results, structured content) may be non-str
            import json
            content = json.dumps(content, ensure_ascii=False)
        total_chars += len(content)
        # role names add a small fixed overhead per message
        total_chars += len(m.get("role", ""))
    return int(total_chars * 0.6)
