"""OpenAI-compatible adapter (identity transform).

Covers any provider that speaks the OpenAI Chat Completions API:
  - SiliconFlow (api.siliconflow.cn/v1)
  - DeepSeek official (api.deepseek.com/v1)
  - OpenAI direct (api.openai.com/v1)
  - Google Gemini OpenAI-compat shim (generativelanguage.googleapis.com/v1beta/openai/)
  - Local runtimes: ollama, vllm (OpenAI-compatible mode)

transform_request is identity (openai format is the native format);
transform_response normalizes the openai ChatCompletion into our ModelResponse
and computes cost from usage × cost_per_1k_*.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from utils.llm_providers.base import (
    Choice,
    LLMRequest,
    ModelResponse,
)


class OpenAICompatibleConfig:
    """Identity adapter for any OpenAI-compatible endpoint."""

    def transform_request(self, req: LLMRequest) -> dict:
        """LLMRequest -> kwargs for client.chat.completions.create."""
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.tools is not None:
            kwargs["tools"] = req.tools
        if req.response_format is not None:
            kwargs["response_format"] = req.response_format
        return kwargs

    def transform_response(
        self,
        raw: Any,
        model: str,
        preset_id: str,
        elapsed_ms: float,
        cost_per_1k_input_tokens: float,
        cost_per_1k_output_tokens: float,
    ) -> ModelResponse:
        """openai ChatCompletion -> normalized ModelResponse."""
        choices = [
            Choice(message=c.message, finish_reason=c.finish_reason or "stop")
            for c in (raw.choices or [])
        ]
        usage = {}
        if getattr(raw, "usage", None):
            usage = {
                "prompt_tokens": getattr(raw.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(raw.usage, "completion_tokens", 0) or 0,
            }

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost_usd = (
            prompt_tokens * cost_per_1k_input_tokens / 1000.0
            + completion_tokens * cost_per_1k_output_tokens / 1000.0
        )

        return ModelResponse(
            choices=choices,
            model=getattr(raw, "model", model) or model,
            preset_id=preset_id,
            usage=usage,
            elapsed_ms=elapsed_ms,
            cost_usd=cost_usd,
            raw=raw,
        )

    def make_client(self, base_url: str, api_key: str) -> OpenAI:
        return OpenAI(base_url=base_url, api_key=api_key)


# Self-register on import so `import utils.llm_providers.openai_compat`
# makes "openai_compat" available in ADAPTER_REGISTRY.
from utils.llm_providers.base import register_adapter  # noqa: E402

register_adapter("openai_compat", OpenAICompatibleConfig)
