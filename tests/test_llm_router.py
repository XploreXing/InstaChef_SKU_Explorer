"""Tests for LLMRouter: preset lookup, fallback, cooldown, metrics."""

import pytest
from openai import APITimeoutError, BadRequestError, RateLimitError
from unittest.mock import MagicMock

from utils.llm_router import LLMRouter, CallMetric
from utils.llm_circuit_breaker import CircuitBreaker


# ── fixtures ────────────────────────────────────────────────────────────────
def _preset_cfg(presets):
    return {
        "presets": presets,
        "cooldown_seconds": 60,
        "default_preset": None,
    }


PRESETS = [
    {"id": "p1", "label": "P1", "base_url": "http://a", "api_key_env": "K1",
     "adapter": "openai_compat", "models": {"generator": "m1"},
     "cost_per_1k_input_tokens": 1.0, "cost_per_1k_output_tokens": 2.0},
    {"id": "p2", "label": "P2", "base_url": "http://b", "api_key_env": "K2",
     "adapter": "openai_compat", "models": {"generator": "m2"},
     "cost_per_1k_input_tokens": 3.0, "cost_per_1k_output_tokens": 4.0},
]


def _mock_handler(resp=None, exc=None):
    """Build a handler whose .chat returns resp or raises exc, once per preset."""
    h = MagicMock()
    if exc is not None:
        h.chat.side_effect = exc
    elif resp is not None:
        h.chat.return_value = resp
    return h


def _mock_response(model="m1", preset_id="p1", in_tok=10, out_tok=5, cost=0.02):
    r = MagicMock()
    r.model = model
    r.preset_id = preset_id
    r.usage = {"prompt_tokens": in_tok, "completion_tokens": out_tok}
    r.cost_usd = cost
    r.elapsed_ms = 100.0
    return r


# ── preset resolution ─────────────────────────────────────────────────────────
def test_selected_preset_only_used():
    """User-chosen preset is used alone; no fallback even on failure."""
    cfg = _preset_cfg(PRESETS)
    exc = APITimeout("boom")
    router = LLMRouter(cfg, role="generator", selected_preset_id="p2",
                      handler=_mock_handler(exc=exc))
    with pytest.raises(APITimeout):
        router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    # only p2 was attempted
    assert len(router.metrics) == 1
    assert router.metrics[0].preset_id == "p2"
    assert not router.metrics[0].success


def test_no_selected_preset_uses_fallback_chain():
    cfg = _preset_cfg(PRESETS)
    resp = _mock_response(model="m2", preset_id="p2")
    # first preset times out, second succeeds
    handler = MagicMock()
    handler.chat.side_effect = [APITimeout("boom"), resp]
    router = LLMRouter(cfg, role="generator", handler=handler)
    r = router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    assert r.preset_id == "p2"
    assert len(router.metrics) == 2
    assert not router.metrics[0].success  # p1 failed
    assert router.metrics[1].success  # p2 succeeded


def test_fallback_on_rate_limit():
    cfg = _preset_cfg(PRESETS)
    resp = _mock_response(preset_id="p2")
    rl = RateLimitError("rl", response=MagicMock(status_code=429, headers={}), body="rl")
    handler = MagicMock()
    handler.chat.side_effect = [rl, resp]
    router = LLMRouter(cfg, role="generator", handler=handler)
    r = router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    assert r.preset_id == "p2"


def test_fallback_on_prompt_too_long():
    cfg = _preset_cfg(PRESETS)
    resp = _mock_response(preset_id="p2")
    br = _bad_request("prompt is too long")
    handler = MagicMock()
    handler.chat.side_effect = [br, resp]
    router = LLMRouter(cfg, role="generator", handler=handler)
    r = router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    assert r.preset_id == "p2"


def test_no_fallback_on_plain_400():
    cfg = _preset_cfg(PRESETS)
    br = _bad_request("invalid model: foo")
    handler = MagicMock()
    handler.chat.side_effect = br
    router = LLMRouter(cfg, role="generator", handler=handler)
    with pytest.raises(BadRequestError):
        router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    # only first preset tried
    assert len(router.metrics) == 1
    assert router.metrics[0].preset_id == "p1"


def test_all_providers_fail_reraises_last():
    cfg = _preset_cfg(PRESETS)
    handler = MagicMock()
    handler.chat.side_effect = [APITimeout("a"), APITimeout("b")]
    router = LLMRouter(cfg, role="generator", handler=handler)
    with pytest.raises(APITimeout):
        router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    assert len(router.metrics) == 2
    assert all(not m.success for m in router.metrics)


def test_cooldown_skips_failed_preset():
    """After p1 fails and is cooled down, next call skips p1 -> uses p2."""
    cfg = _preset_cfg(PRESETS)
    breaker = CircuitBreaker(cooldown_seconds=60)
    # mark p1 cooled down manually (simulating prior failure)
    breaker.record_failure("p1")

    resp = _mock_response(preset_id="p2")
    handler = MagicMock()
    handler.chat.return_value = resp
    router = LLMRouter(cfg, role="generator", breaker=breaker, handler=handler)
    r = router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    assert r.preset_id == "p2"
    # handler called once (p1 skipped), not twice
    assert handler.chat.call_count == 1
    assert handler.chat.call_args[0][1].id == "p2"


def test_selected_preset_ignores_cooldown():
    """User explicitly chose p1 even though it's cooled down — honor it."""
    cfg = _preset_cfg(PRESETS)
    breaker = CircuitBreaker(cooldown_seconds=60)
    breaker.record_failure("p1")
    resp = _mock_response(preset_id="p1")
    handler = MagicMock()
    handler.chat.return_value = resp
    router = LLMRouter(cfg, role="generator", selected_preset_id="p1",
                      breaker=breaker, handler=handler)
    r = router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    assert r.preset_id == "p1"
    assert handler.chat.call_count == 1


def test_metrics_record_cost_and_tokens():
    cfg = _preset_cfg(PRESETS)
    resp = _mock_response(model="m1", preset_id="p1", in_tok=100, out_tok=50, cost=0.5)
    router = LLMRouter(cfg, role="generator", selected_preset_id="p1",
                      handler=_mock_handler(resp=resp))
    router.chat(messages=[{"role": "user", "content": "hi"}],
                temperature=0, max_tokens=10)
    m = router.recent_metrics()[0]
    assert m.preset_id == "p1"
    assert m.model == "m1"
    assert m.role == "generator"
    assert m.prompt_tokens == 100
    assert m.completion_tokens == 50
    assert m.cost_usd == 0.5
    assert m.success is True


def test_role_default_model_from_preset():
    cfg = _preset_cfg([{
        "id": "p1", "label": "P1", "base_url": "http://a", "api_key_env": "K1",
        "adapter": "openai_compat",
        "models": {"generator": "m-gen", "evaluator": "m-eval"},
    }])
    resp = _mock_response()
    handler = MagicMock()
    handler.chat.return_value = resp
    router = LLMRouter(cfg, role="evaluator", selected_preset_id="p1", handler=handler)
    router.chat(messages=[{"role": "user", "content": "hi"}],
                temperature=0, max_tokens=10)
    req = handler.chat.call_args[0][2]
    assert req.model == "m-eval"


def test_model_override_takes_precedence():
    cfg = _preset_cfg(PRESETS)
    resp = _mock_response()
    handler = MagicMock()
    handler.chat.return_value = resp
    router = LLMRouter(cfg, role="generator", selected_preset_id="p1", handler=handler)
    router.chat(messages=[{"role": "user", "content": "hi"}],
                temperature=0, max_tokens=10, model="custom-model")
    req = handler.chat.call_args[0][2]
    assert req.model == "custom-model"


# ── thinking switch ───────────────────────────────────────────────────────────
THINKING_PRESET = {
    "id": "p1", "label": "P1", "base_url": "http://a", "api_key_env": "K1",
    "adapter": "openai_compat", "models": {"generator": "m1"},
    "thinking_extra_body": {
        "enabled": {"thinking": {"type": "enabled"}},
        "disabled": {"thinking": {"type": "disabled"}},
    },
}


def _request_sent(preset, **chat_kwargs):
    """The LLMRequest the router hands to the HTTP handler for one chat call."""
    handler = MagicMock()
    handler.chat.return_value = _mock_response()
    router = LLMRouter(_preset_cfg([preset]), role="generator",
                       selected_preset_id="p1", handler=handler)
    router.chat(messages=[{"role": "user", "content": "hi"}],
                temperature=0, max_tokens=10, **chat_kwargs)
    return handler.chat.call_args[0][2]


@pytest.mark.parametrize("thinking, expected", [
    (False, {"thinking": {"type": "disabled"}}),
    (True, {"thinking": {"type": "enabled"}}),
    (None, None),  # caller has no preference -> endpoint default
])
def test_thinking_flag_maps_to_preset_extra_body(thinking, expected):
    assert _request_sent(THINKING_PRESET, thinking=thinking).extra_body == expected


def test_thinking_flag_ignored_when_preset_declares_no_switch():
    """An endpoint that cannot switch thinking keeps its default."""
    assert _request_sent(PRESETS[0], thinking=False).extra_body is None


@pytest.mark.parametrize("preset_id", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_official_deepseek_presets_can_switch_thinking(preset_id):
    """The official V4 models think by default and reasoning tokens count
    toward max_tokens, so config.yaml must say how to turn it off."""
    import yaml
    from pathlib import Path

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    llm_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["llm"]
    preset = next(p for p in LLMRouter._build_presets(llm_cfg) if p.id == preset_id)

    assert preset.extra_body_for(thinking=False) == {"thinking": {"type": "disabled"}}
    assert preset.extra_body_for(thinking=True) == {"thinking": {"type": "enabled"}}


def test_backward_compat_flat_config():
    """No 'presets' key -> synthesize single preset from flat fields."""
    cfg = {
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "LLM_API_KEY",
        "generator_model": "deepseek-ai/DeepSeek-V3",
        "evaluator_model": "deepseek-ai/DeepSeek-V3",
    }
    resp = _mock_response()
    router = LLMRouter(cfg, role="generator", handler=_mock_handler(resp=resp))
    assert len(router.presets) == 1
    assert router.presets[0].id == "legacy"
    r = router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    assert r is resp


def test_unknown_selected_preset_raises():
    cfg = _preset_cfg(PRESETS)
    router = LLMRouter(cfg, role="generator", selected_preset_id="nope",
                      handler=_mock_handler(resp=_mock_response()))
    with pytest.raises(KeyError, match="not in presets"):
        router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)


def test_selected_preset_with_no_key_raises_valueerror(monkeypatch):
    """Missing API key -> ValueError from the real HTTP handler (not the
    mock handler, which would bypass the key check)."""
    cfg = _preset_cfg(PRESETS)
    monkeypatch.delenv("K1", raising=False)
    monkeypatch.delenv("K2", raising=False)
    # use the REAL BaseLLMHTTPHandler so the key check fires
    from utils.llm_providers.base import BaseLLMHTTPHandler
    router = LLMRouter(cfg, role="generator", selected_preset_id="p1",
                      handler=BaseLLMHTTPHandler())
    with pytest.raises(ValueError, match="no API key"):
        router.chat(messages=[{"role": "user", "content": "hi"}],
                    temperature=0, max_tokens=10)
    # failure recorded as a metric
    assert router.metrics and not router.metrics[0].success


# ── helpers ─────────────────────────────────────────────────────────────────
def _bad_request(body: str):
    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {}
    return BadRequestError(body, response=resp, body=body)


class APITimeout(APITimeoutError):
    """Concrete subclass to bypass openai SDK's __init__ signature (which
    requires a request arg in newer versions)."""
    def __init__(self, message: str):
        Exception.__init__(self, message)
