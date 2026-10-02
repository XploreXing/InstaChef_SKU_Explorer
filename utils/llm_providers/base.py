"""Base translation layer: dataclasses, error predicates, and the
provider-agnostic HTTP handler.

This module is the single source of truth for:
  - prompt-too-long detection (migrated here from agents/generator.py so all
    providers benefit; agents/generator.py re-exports it for test compat)
  - fallback eligibility (which errors should trigger a provider switch)
  - the normalized request/response shapes every provider must produce
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

# ── prompt-too-long detection ──────────────────────────────────────────────
# Markers expected across OpenAI-compatible endpoints (incl. SiliconFlow /
# DeepSeek). 措辞由端点决定、不稳定，故用 isinstance(BadRequestError) 收窄在前、
# 软匹配关键词兜底在后：BadRequestError 跨端点跨模型稳定（任何 4xx 都映射进来），
# 先排除 401/429/529，再在 400 里匹配措辞。命中不了只是漏判（值=跟没防御一样），
# 不会误判为 too-long 而错误地 compact。
_PROMPT_TOO_LONG_MARKERS = (
    "context length",
    "maximum context",
    "too long",
    "context_window",
    "context length exceeded",
)


def is_prompt_too_long_error(e: Exception) -> bool:
    """Check whether an API error indicates prompt/context too long."""
    if not isinstance(e, BadRequestError):
        return False
    msg = str(e).lower()
    return any(marker in msg for marker in _PROMPT_TOO_LONG_MARKERS)


def is_fallback_eligible(e: Exception) -> bool:
    """Whether `e` should trigger a switch to the next provider/preset.

    Fallback-eligible: transient/availability failures (the provider is
    temporarily or permanently unusable, another might work). Non-eligible
    errors (plain 400 malformed, etc.) raise immediately — retrying on a
    different provider won't fix a structurally invalid request.
    """
    if isinstance(
        e, (APITimeoutError, APIConnectionError, AuthenticationError, RateLimitError)
    ):
        return True
    if isinstance(e, APIStatusError) and getattr(e, "status_code", 0) >= 500:
        return True
    if isinstance(e, BadRequestError) and is_prompt_too_long_error(e):
        return True
    return False


# ── normalized data shapes ─────────────────────────────────────────────────
@dataclass
class PresetConfig:
    """A fully-resolved model preset (one entry in config.llm.presets).

    `adapter` explicitly names the translation path ("openai_compat" for
    identity / SiliconFlow / DeepSeek / Gemini-OpenAI-shim / ollama / vllm,
    "anthropic" for Claude native API). This explicit field replaces
    litellm's model-string inference (get_llm_provider).
    """

    id: str
    label: str
    base_url: str
    api_key_env: str
    adapter: str  # "openai_compat" | "anthropic"
    models: dict[str, str] = field(default_factory=dict)  # role -> model name
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    api_key: str = ""  # resolved at runtime from api_key_env
    # Request-body fields this endpoint uses to switch thinking, keyed by
    # "enabled" / "disabled". Empty = its default cannot be changed.
    thinking_extra_body: dict[str, dict] = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """Resolve the API key from the explicit value or the env var."""
        return self.api_key or os.getenv(self.api_key_env, "")

    def extra_body_for(self, thinking: bool | None) -> dict | None:
        """Body fields that switch thinking as asked. None when the caller has
        no preference or this endpoint declares no switch for it."""
        if thinking is None:
            return None
        return self.thinking_extra_body.get("enabled" if thinking else "disabled")

    def model_for(self, role: str) -> str:
        """Return the model name for a given role (generator/evaluator/...)."""
        return self.models.get(role, "")


@dataclass
class LLMRequest:
    """Provider-agnostic request. Providers' transform_request converts this
    into their native request body."""

    messages: list[dict]
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    tools: list[dict] | None = None
    response_format: dict | None = None
    extra_body: dict | None = None  # provider-specific body fields, sent as-is


@dataclass
class Choice:
    """One choice in a normalized ModelResponse. Mirrors the openai Choice
    shape so existing agent code (response.choices[0].message / .finish_reason)
    works unchanged across providers."""

    message: Any  # openai ChatCompletionMessage-like (has .content, .tool_calls)
    finish_reason: str = "stop"


@dataclass
class ModelResponse:
    """Normalized response (inspired by litellm's ModelResponse). All
    providers' transform_response must produce this shape so callers see a
    uniform interface regardless of the underlying provider SDK."""

    choices: list[Choice]
    model: str
    preset_id: str
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens
    elapsed_ms: float = 0.0
    cost_usd: float = 0.0
    raw: Any = None  # original SDK response, for advanced introspection


# ── provider config protocol ──────────────────────────────────────────────
class BaseProviderConfig(Protocol):
    """Each provider implements these three methods. The HTTP handler calls
    them in order: make_client -> transform_request -> (send) ->
    transform_response. Adding a provider = implementing this Protocol; the
    handler itself never changes (litellm BaseConfig pattern)."""

    def transform_request(self, req: LLMRequest) -> dict: ...

    def transform_response(
        self,
        raw: Any,
        model: str,
        preset_id: str,
        elapsed_ms: float,
        cost_per_1k_input_tokens: float,
        cost_per_1k_output_tokens: float,
    ) -> ModelResponse: ...

    def make_client(self, base_url: str, api_key: str) -> Any: ...


# ── adapter registry ───────────────────────────────────────────────────────
# Explicit adapter-name -> Config-class mapping. PresetConfig.adapter names
# the translation path; the HTTP handler instantiates the Config from this
# registry. This replaces litellm's model-string inference (get_llm_provider):
# presets carry their own provider identity, no reverse-inference needed.
# Populated lazily (openai_compat is the only default-registered adapter;
# anthropic is registered by importers that need it, since it pulls an
# optional dependency). The default registry is mutated via
# `register_adapter` rather than hard-importing anthropic here, to keep
# `import utils.llm_providers` dependency-light.
ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(name: str, config_cls: type) -> None:
    """Register an adapter class under `name`. Call at import time of an
    adapter module (e.g. openai_compat registers itself)."""
    ADAPTER_REGISTRY[name] = config_cls


def get_adapter(name: str) -> Any:
    """Look up an adapter config class by name. Raises KeyError with a clear
    message if unknown."""
    if name not in ADAPTER_REGISTRY:
        raise KeyError(
            f"Unknown adapter '{name}'. Registered: {list(ADAPTER_REGISTRY)}. "
            f"Did you import the adapter module (e.g. "
            f"'from utils.llm_providers import openai_compat')?"
        )
    return ADAPTER_REGISTRY[name]


# ── provider-agnostic HTTP handler ────────────────────────────────────────
class BaseLLMHTTPHandler:
    """Sends a request via a provider config. Provider-agnostic: it delegates
    all format conversion to the provider's transform_* methods.

    Exceptions are raised as-is; the LLMRouter layer decides whether to
    fall back to another preset based on is_fallback_eligible().
    """

    def chat(self, provider: BaseProviderConfig, preset: PresetConfig, req: LLMRequest) -> ModelResponse:
        api_key = preset.resolve_api_key()
        if not api_key:
            raise ValueError(
                f"Preset '{preset.id}' has no API key "
                f"(env {preset.api_key_env} unset, no override)"
            )
        client = provider.make_client(preset.base_url, api_key)
        kwargs = provider.transform_request(req)

        t0 = time.monotonic()
        raw = client.chat.completions.create(**kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return provider.transform_response(
            raw,
            model=req.model,
            preset_id=preset.id,
            elapsed_ms=elapsed_ms,
            cost_per_1k_input_tokens=preset.cost_per_1k_input_tokens,
            cost_per_1k_output_tokens=preset.cost_per_1k_output_tokens,
        )
