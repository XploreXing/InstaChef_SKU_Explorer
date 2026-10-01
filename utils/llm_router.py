"""LLMRouter: preset lookup, fallback orchestration, cost/latency tracing.

RESPONSIBILITIES
  1. Resolve preset id -> PresetConfig (explicit lookup, no model-string
     inference — unlike litellm's get_llm_provider).
  2. Decide attempt order via a RoutingStrategy (default: sequential fallback
     skipping cooled-down presets).
  3. Call BaseLLMHTTPHandler.chat() on each in order; on fallback-eligible
     errors, mark the preset failed (breaker) and try the next.
  4. Record CallMetric for every attempt (preset, model, latency, tokens,
     cost, success/error) for observability.

FALLBACK SEMANTICS
  - If `selected_preset_id` is set (user chose a preset in the UI): ONLY that
    preset is tried. A user's explicit choice is intent — don't silently
    switch models on them. Failure raises.
  - If `selected_preset_id` is None (no choice / default / ops mode): try
    presets in config order, falling back on transient errors. This is the
    "default preset + fallback chain" mode.

BREAKER INJECTION
  The router accepts an external CircuitBreaker so Streamlit can share one
  process-wide breaker (@st.cache_resource) across reruns. If none is passed,
  the router creates its own (unit-test / non-Streamlit use).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from utils.llm_circuit_breaker import CircuitBreaker
from utils.llm_providers.base import (
    BaseLLMHTTPHandler,
    LLMRequest,
    ModelResponse,
    PresetConfig,
    get_adapter,
    is_fallback_eligible,
)
from utils.llm_token_counter import estimate_tokens


@dataclass
class CallMetric:
    """One recorded LLM call attempt (success or failure)."""

    preset_id: str
    model: str
    role: str
    elapsed_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    success: bool
    error_type: str = ""


class RoutingStrategy(Protocol):
    """Decides the order in which presets are attempted for a call."""

    def select(
        self,
        presets: list[PresetConfig],
        breaker: CircuitBreaker,
    ) -> list[PresetConfig]:
        ...


class SequentialFallbackStrategy:
    """Default strategy: config order, skipping cooled-down presets.

    If only one preset is selected (user chose), returns just that one
    (regardless of cooldown — an explicit choice is honored even if recently
    failed; the breaker still records the failure for observability).
    """

    def select(
        self,
        presets: list[PresetConfig],
        breaker: CircuitBreaker,
    ) -> list[PresetConfig]:
        if len(presets) == 1:
            return list(presets)
        return [p for p in presets if not breaker.is_cooled_down(p.id)]


class LLMRouter:
    """Routes LLM calls across presets with fallback + tracing."""

    def __init__(
        self,
        llm_cfg: dict,
        role: str,
        selected_preset_id: str | None = None,
        breaker: CircuitBreaker | None = None,
        handler: BaseLLMHTTPHandler | None = None,
    ):
        self.role = role
        self.llm_cfg = llm_cfg
        self.selected_preset_id = selected_preset_id or llm_cfg.get("default_preset")

        self.presets = self._build_presets(llm_cfg)
        self.breaker = breaker or CircuitBreaker(
            cooldown_seconds=llm_cfg.get("cooldown_seconds", 60)
        )
        self.strategy = SequentialFallbackStrategy()
        self.handler = handler or BaseLLMHTTPHandler()
        self.metrics: list[CallMetric] = []

    # ── preset resolution ──────────────────────────────────────────────────
    @staticmethod
    def _build_presets(llm_cfg: dict) -> list[PresetConfig]:
        """Parse config into PresetConfig list.

        Supports two shapes:
          - New: llm_cfg["presets"] -> list of preset dicts
          - Legacy (backward compat): flat base_url/api_key_env/{role}_model
            -> single synthesized preset
        """
        if "presets" in llm_cfg and llm_cfg["presets"]:
            return [LLMRouter._parse_preset(p) for p in llm_cfg["presets"]]
        # Legacy flat config
        return [LLMRouter._legacy_preset(llm_cfg)]

    @staticmethod
    def _parse_preset(p: dict) -> PresetConfig:
        return PresetConfig(
            id=p["id"],
            label=p.get("label", p["id"]),
            base_url=p["base_url"],
            api_key_env=p.get("api_key_env", ""),
            adapter=p.get("adapter", "openai_compat"),
            models=dict(p.get("models", {})),
            cost_per_1k_input_tokens=p.get("cost_per_1k_input_tokens", 0.0),
            cost_per_1k_output_tokens=p.get("cost_per_1k_output_tokens", 0.0),
            api_key=p.get("api_key", ""),
        )

    @staticmethod
    def _legacy_preset(llm_cfg: dict) -> PresetConfig:
        return PresetConfig(
            id="legacy",
            label=llm_cfg.get("label", "Legacy (flat config)"),
            base_url=llm_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
            api_key_env=llm_cfg.get("api_key_env", "LLM_API_KEY"),
            adapter="openai_compat",
            models={
                "generator": llm_cfg.get("generator_model", ""),
                "evaluator": llm_cfg.get("evaluator_model", ""),
                "enrichment": llm_cfg.get("enrichment_model", ""),
                "summary": llm_cfg.get("generator_model", ""),
            },
        )

    def _active_presets(self) -> list[PresetConfig]:
        """Presets to try for this call: just the selected one, or all."""
        if self.selected_preset_id:
            for p in self.presets:
                if p.id == self.selected_preset_id:
                    return [p]
            raise KeyError(
                f"selected_preset_id '{self.selected_preset_id}' not in presets "
                f"{[p.id for p in self.presets]}"
            )
        return list(self.presets)

    # ── main entry ─────────────────────────────────────────────────────────
    def chat(
        self,
        *,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        model: str | None = None,
    ) -> ModelResponse:
        """Send a chat request, with fallback across eligible presets.

        `model` overrides the preset's role-default model (used by the legacy
        override path where app.py passes a specific model name).
        """
        order = self.strategy.select(self._active_presets(), self.breaker)
        if not order:
            raise RuntimeError(
                f"No available presets to try for role='{self.role}' "
                f"(all cooled down or none configured)"
            )

        last_exc: Exception | None = None
        for preset in order:
            resolved_model = model or preset.model_for(self.role)
            if not resolved_model:
                last_exc = ValueError(
                    f"Preset '{preset.id}' has no model for role '{self.role}'"
                )
                self._record_failure_metric(preset, resolved_model or "?", last_exc)
                continue

            req = LLMRequest(
                messages=messages,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                response_format=response_format,
            )
            try:
                adapter_cls = get_adapter(preset.adapter)
                provider = adapter_cls()
                resp = self.handler.chat(provider, preset, req)
                self.breaker.record_success(preset.id)
                self._record_success_metric(preset, resp)
                return resp
            except Exception as e:
                last_exc = e
                self._record_failure_metric(preset, resolved_model, e)
                if is_fallback_eligible(e) and len(order) > 1:
                    self.breaker.record_failure(preset.id)
                    print(
                        f"[llm] preset '{preset.id}' failed ({type(e).__name__}: {e}); "
                        f"falling back to next preset",
                        flush=True,
                    )
                    continue
                # non-eligible error, or single preset — raise immediately
                raise

        # All presets exhausted (only reachable when multiple presets all
        # failed with fallback-eligible errors)
        if last_exc:
            raise last_exc
        raise RuntimeError("No preset succeeded")

    # ── metrics ────────────────────────────────────────────────────────────
    def _record_success_metric(self, preset: PresetConfig, resp: ModelResponse) -> None:
        self.metrics.append(CallMetric(
            preset_id=preset.id,
            model=resp.model,
            role=self.role,
            elapsed_ms=resp.elapsed_ms,
            prompt_tokens=resp.usage.get("prompt_tokens", 0),
            completion_tokens=resp.usage.get("completion_tokens", 0),
            cost_usd=resp.cost_usd,
            success=True,
        ))

    def _record_failure_metric(self, preset: PresetConfig, model: str, e: Exception) -> None:
        # On failure, estimate prompt tokens from messages for observability
        self.metrics.append(CallMetric(
            preset_id=preset.id,
            model=model,
            role=self.role,
            elapsed_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            success=False,
            error_type=type(e).__name__,
        ))

    def recent_metrics(self) -> list[CallMetric]:
        return list(self.metrics)

    def flush_metrics(self) -> list[CallMetric]:
        m = self.metrics
        self.metrics = []
        return m
