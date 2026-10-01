"""LLM provider abstraction layer.

Translation-layer architecture (inspired by litellm's BaseConfig pattern):
each provider implements transform_request / transform_response so the HTTP
handler stays provider-agnostic. Requests/responses are normalized to/from
an OpenAI-shaped LLMRequest/ModelResponse so the generator/evaluator agents
(which use the openai SDK style) need no changes when swapping providers.

Adapter selection is explicit (PresetConfig.adapter field) rather than
inferred from model strings (litellm's get_llm_provider) — presets carry
their own provider identity, suiting closed-configuration internal tools.

Importing this package registers the built-in adapters
(openai_compat, anthropic) in ADAPTER_REGISTRY.
"""

from utils.llm_providers.base import (
    ADAPTER_REGISTRY,
    BaseLLMHTTPHandler,
    BaseProviderConfig,
    LLMRequest,
    ModelResponse,
    PresetConfig,
    get_adapter,
    is_fallback_eligible,
    is_prompt_too_long_error,
    register_adapter,
)
from utils.llm_providers.openai_compat import OpenAICompatibleConfig
from utils.llm_providers.anthropic import AnthropicConfig  # noqa: F401  (registers adapter)

__all__ = [
    "ADAPTER_REGISTRY",
    "BaseLLMHTTPHandler",
    "BaseProviderConfig",
    "OpenAICompatibleConfig",
    "AnthropicConfig",
    "LLMRequest",
    "ModelResponse",
    "PresetConfig",
    "get_adapter",
    "register_adapter",
    "is_fallback_eligible",
    "is_prompt_too_long_error",
]
