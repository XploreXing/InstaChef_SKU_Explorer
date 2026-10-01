"""Anthropic adapter: OpenAI <-> Claude native translation layer.

WHY THIS EXISTS
The generator/evaluator agents are written in openai-SDK style:
    response = client.chat.completions.create(messages=[...], tools=SCHEMAS)
    msg = response.choices[0].message
    if msg.tool_calls: ...
Claude's native API does NOT speak this shape — system prompts live in a
top-level param (not in messages), tool schemas use {name, input_schema}
rather than {type:"function", function:{...}}, and responses are content
blocks (text / tool_use) rather than choices. Calling Claude directly from
the agents would force agent code to branch per provider.

This adapter bridges that: transform_request converts the openai-shaped
LLMRequest into a Claude-native request body, and transform_response
converts Claude's content blocks back into the openai-shaped ModelResponse.
Agent code stays provider-agnostic.

STATUS (v1)
Translation is implemented and unit-tested, but NOT wired into the default
preset chain — the anthropic SDK is an optional dependency. Selecting an
anthropic-adapter preset requires `pip install anthropic`; the preset's
make_client imports it lazily so the module loads without it installed.

The translation handles:
  - system prompt extraction (role=system -> top-level `system`)
  - tool schema conversion (openai function-calling -> anthropic tools)
  - response_format json_object -> prompt nudge (Claude has no json mode;
    we rely on the prompt nudge already present in agent code + the tools)
  - tool result messages (role=tool -> role=user with tool_result content)
  - response content blocks (text/tool_use -> choices[0].message + tool_calls)
"""

from __future__ import annotations

import json
from typing import Any

from utils.llm_providers.base import Choice, LLMRequest, ModelResponse


class AnthropicConfig:
    """Translates OpenAI-shaped requests to Claude native API and back."""

    # ── request: openai -> anthropic ──────────────────────────────────────
    def transform_request(self, req: LLMRequest) -> dict:
        """LLMRequest (openai shape) -> anthropic messages.create kwargs."""
        system_text, messages = self._split_system(req.messages)

        body: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if system_text:
            body["system"] = system_text
        if req.tools is not None:
            body["tools"] = [self._convert_tool_schema(t) for t in req.tools]
        # Note: response_format json_object has no Claude equivalent. The
        # agents already include a "Output ONLY JSON..." prompt nudge, so we
        # rely on that. We deliberately do NOT inject tool_forcing here to
        # avoid changing agent semantics.
        return body

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        """Pull role=system entries out of messages (Claude requires system
        as a top-level param, not inside messages)."""
        system_parts: list[str] = []
        rest: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    system_parts.append(json.dumps(content, ensure_ascii=False))
            elif m.get("role") == "tool":
                # openai tool result -> anthropic user message with tool_result block
                rest.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": m.get("content", ""),
                    }],
                })
            else:
                rest.append(m)
        return "\n\n".join(s for s in system_parts if s), rest

    @staticmethod
    def _convert_tool_schema(openai_tool: dict) -> dict:
        """openai {type:function, function:{name, parameters}} ->
        anthropic {name, description, input_schema}."""
        fn = openai_tool.get("function", openai_tool)
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        }

    # ── response: anthropic -> openai shape ────────────────────────────────
    def transform_response(
        self,
        raw: Any,
        model: str,
        preset_id: str,
        elapsed_ms: float,
        cost_per_1k_input_tokens: float,
        cost_per_1k_output_tokens: float,
    ) -> ModelResponse:
        """anthropic response (content blocks) -> openai-shaped ModelResponse."""
        content_blocks = getattr(raw, "content", []) or []
        text_parts: list[str] = []
        tool_calls: list[Any] = []
        finish_reason = "stop"

        for block in content_blocks:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if btype == "text":
                txt = getattr(block, "text", None) or (block.get("text", "") if isinstance(block, dict) else "")
                text_parts.append(txt)
            elif btype == "tool_use":
                tool_calls.append(self._convert_tool_use(block))

        stop_reason = getattr(raw, "stop_reason", None)
        if stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif stop_reason == "max_tokens":
            finish_reason = "length"

        message = _AnthropicMessage(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        usage = {}
        in_tok = 0
        out_tok = 0
        if getattr(raw, "usage", None):
            in_tok = getattr(raw.usage, "input_tokens", 0) or 0
            out_tok = getattr(raw.usage, "output_tokens", 0) or 0
            usage = {"prompt_tokens": in_tok, "completion_tokens": out_tok}
        cost_usd = (
            in_tok * cost_per_1k_input_tokens / 1000.0
            + out_tok * cost_per_1k_output_tokens / 1000.0
        )

        return ModelResponse(
            choices=[Choice(message=message, finish_reason=finish_reason)],
            model=getattr(raw, "model", model) or model,
            preset_id=preset_id,
            usage=usage,
            elapsed_ms=elapsed_ms,
            cost_usd=cost_usd,
            raw=raw,
        )

    @staticmethod
    def _convert_tool_use(block: Any) -> dict:
        """anthropic tool_use block -> openai tool_call dict (message.tool_calls)."""
        if isinstance(block, dict):
            return {
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            }
        return {
            "id": getattr(block, "id", ""),
            "type": "function",
            "function": {
                "name": getattr(block, "name", ""),
                "arguments": json.dumps(getattr(block, "input", {}), ensure_ascii=False),
            },
        }

    # ── client ─────────────────────────────────────────────────────────────
    def make_client(self, base_url: str, api_key: str) -> Any:
        """Lazily import anthropic SDK. Module loads without anthropic
        installed; only selecting an anthropic preset triggers this import."""
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicConfig preset selected but the 'anthropic' package is not "
                "installed. Install it with: pip install anthropic"
            ) from e
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return Anthropic(**kwargs)


class _AnthropicMessage:
    """Mimics openai ChatCompletionMessage enough for agent code:
    agents access .content and .tool_calls."""

    def __init__(self, content: str | None, tool_calls: list | None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


# Self-register on import. Importing this module registers the "anthropic"
# adapter; the anthropic SDK itself is only imported lazily inside
# make_client, so importing this module does NOT require `anthropic` installed.
from utils.llm_providers.base import register_adapter  # noqa: E402

register_adapter("anthropic", AnthropicConfig)
