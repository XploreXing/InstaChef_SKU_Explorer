"""Tests for the LLM provider translation layer.

These mirror litellm's translation-unit-test pattern: each transform is
verified in isolation, without making real API calls.
"""

from unittest.mock import MagicMock

import pytest
from openai import BadRequestError

from utils.llm_providers.base import (
    LLMRequest,
    ModelResponse,
    PresetConfig,
    is_fallback_eligible,
    is_prompt_too_long_error,
)
from utils.llm_providers.openai_compat import OpenAICompatibleConfig
from utils.llm_providers.anthropic import AnthropicConfig, _AnthropicMessage


# ── helpers ────────────────────────────────────────────────────────────────
def _openai_chat_completion(
    content="hi", tool_calls=None, finish_reason="stop",
    prompt_tokens=10, completion_tokens=5, model="deepseek-v4-flash",
):
    """Build a fake openai.ChatCompletion object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    raw = MagicMock()
    raw.choices = [choice]
    raw.usage = usage
    raw.model = model
    return raw


def _bad_request(body: str):
    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {}
    return BadRequestError(body, response=resp, body=body)


# ── is_prompt_too_long_error ───────────────────────────────────────────────
def test_too_long_detected_on_context_length_phrase():
    e = _bad_request(
        "This model's maximum context length is 8192 tokens. "
        "However, your messages resulted in 12345 tokens."
    )
    assert is_prompt_too_long_error(e) is True


def test_too_long_detected_on_prompt_is_too_long():
    e = _bad_request("Error: prompt is too long")
    assert is_prompt_too_long_error(e) is True


def test_too_long_false_on_plain_400():
    e = _bad_request("invalid model: foo")
    assert is_prompt_too_long_error(e) is False


def test_too_long_false_on_non_badrequest():
    assert is_prompt_too_long_error(ValueError("context length")) is False


# ── is_fallback_eligible ────────────────────────────────────────────────────
def test_fallback_eligible_on_prompt_too_long():
    assert is_fallback_eligible(_bad_request("prompt is too long")) is True


def test_fallback_eligible_on_timeout():
    from openai import APITimeoutError
    assert is_fallback_eligible(APITimeoutError("timeout")) is True


def test_fallback_eligible_on_rate_limit():
    from openai import RateLimitError
    resp = MagicMock(); resp.status_code = 429; resp.headers = {}
    e = RateLimitError("rl", response=resp, body="rl")
    assert is_fallback_eligible(e) is True


def test_fallback_eligible_on_auth_error():
    from openai import AuthenticationError
    resp = MagicMock(); resp.status_code = 401; resp.headers = {}
    e = AuthenticationError("auth", response=resp, body="auth")
    assert is_fallback_eligible(e) is True


def test_fallback_not_eligible_on_plain_400():
    assert is_fallback_eligible(_bad_request("invalid model: foo")) is False


# ── OpenAICompatibleConfig: transform_request identity ─────────────────────
def test_openai_transform_request_identity():
    cfg = OpenAICompatibleConfig()
    req = LLMRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="deepseek-v4-flash",
        temperature=0.7,
        max_tokens=4096,
        tools=[{"type": "function", "function": {"name": "f"}}],
        response_format={"type": "json_object"},
    )
    kwargs = cfg.transform_request(req)
    assert kwargs["model"] == "deepseek-v4-flash"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 4096
    assert kwargs["tools"] == [{"type": "function", "function": {"name": "f"}}]
    assert kwargs["response_format"] == {"type": "json_object"}


def test_openai_transform_request_omits_optional_when_none():
    cfg = OpenAICompatibleConfig()
    req = LLMRequest(
        messages=[], model="m", temperature=0, max_tokens=10,
        tools=None, response_format=None,
    )
    kwargs = cfg.transform_request(req)
    assert "tools" not in kwargs
    assert "response_format" not in kwargs
    assert "extra_body" not in kwargs


def test_openai_transform_request_forwards_extra_body():
    """Provider-specific body fields (e.g. DeepSeek's thinking switch) pass through."""
    cfg = OpenAICompatibleConfig()
    req = LLMRequest(
        messages=[], model="m", temperature=0, max_tokens=10,
        extra_body={"thinking": {"type": "disabled"}},
    )
    kwargs = cfg.transform_request(req)
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


# ── OpenAICompatibleConfig: transform_response normalization ───────────────
def test_openai_transform_response_normalizes():
    cfg = OpenAICompatibleConfig()
    raw = _openai_chat_completion(
        content="hello", prompt_tokens=10, completion_tokens=5,
        model="deepseek-v4-flash",
    )
    resp = cfg.transform_response(
        raw, model="deepseek-v4-flash", preset_id="p1",
        elapsed_ms=123.0,
        cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
    )
    assert isinstance(resp, ModelResponse)
    assert resp.choices[0].message.content == "hello"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert resp.preset_id == "p1"
    assert resp.elapsed_ms == 123.0
    assert resp.model == "deepseek-v4-flash"
    assert resp.raw is raw


def test_openai_transform_response_with_tool_calls():
    cfg = OpenAICompatibleConfig()
    tc = MagicMock(); tc.id = "tc1"; tc.function.name = "f"; tc.function.arguments = "{}"
    raw = _openai_chat_completion(
        content=None, tool_calls=[tc], finish_reason="tool_calls",
    )
    resp = cfg.transform_response(
        raw, model="m", preset_id="p1", elapsed_ms=0,
        cost_per_1k_input_tokens=0, cost_per_1k_output_tokens=0,
    )
    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.choices[0].message.tool_calls == [tc]


# ── cost calculation ────────────────────────────────────────────────────────
def test_cost_calculation():
    cfg = OpenAICompatibleConfig()
    raw = _openai_chat_completion(prompt_tokens=1000, completion_tokens=500)
    resp = cfg.transform_response(
        raw, model="m", preset_id="p1", elapsed_ms=0,
        cost_per_1k_input_tokens=2.0,  # $2 / 1k input
        cost_per_1k_output_tokens=8.0,  # $8 / 1k output
    )
    # 1000 * 2/1000 + 500 * 8/1000 = 2.0 + 4.0 = 6.0
    assert resp.cost_usd == pytest.approx(6.0)


def test_cost_zero_when_no_usage():
    cfg = OpenAICompatibleConfig()
    raw = MagicMock()
    raw.choices = [MagicMock(message=MagicMock(content="x", tool_calls=None), finish_reason="stop")]
    raw.usage = None
    raw.model = "m"
    resp = cfg.transform_response(
        raw, model="m", preset_id="p1", elapsed_ms=0,
        cost_per_1k_input_tokens=2.0, cost_per_1k_output_tokens=8.0,
    )
    assert resp.cost_usd == 0.0
    assert resp.usage == {}


# ── AnthropicConfig: transform_request ─────────────────────────────────────
def test_anthropic_transform_request_extracts_system():
    cfg = AnthropicConfig()
    req = LLMRequest(
        messages=[
            {"role": "system", "content": "you are a chef"},
            {"role": "user", "content": "hi"},
        ],
        model="claude-3-5-sonnet", temperature=0.3, max_tokens=2048,
    )
    body = cfg.transform_request(req)
    assert body["system"] == "you are a chef"
    # system must NOT remain inside messages
    assert all(m["role"] != "system" for m in body["messages"])
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_transform_request_no_system_omits_field():
    cfg = AnthropicConfig()
    req = LLMRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="claude", temperature=0, max_tokens=10,
    )
    body = cfg.transform_request(req)
    assert "system" not in body


def test_anthropic_transform_request_tools():
    cfg = AnthropicConfig()
    openai_tool = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    req = LLMRequest(
        messages=[{"role": "user", "content": "x"}],
        model="claude", temperature=0, max_tokens=10, tools=[openai_tool],
    )
    body = cfg.transform_request(req)
    assert body["tools"][0] == {
        "name": "search",
        "description": "search web",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }


def test_anthropic_transform_request_tool_result_message():
    cfg = AnthropicConfig()
    req = LLMRequest(
        messages=[
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": None, "tool_calls": [MagicMock()]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result data"},
        ],
        model="claude", temperature=0, max_tokens=10,
    )
    body = cfg.transform_request(req)
    # tool message converted to user role with tool_result content block
    tool_msg = body["messages"][-1]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == "tc1"
    assert tool_msg["content"][0]["content"] == "result data"


# ── AnthropicConfig: transform_response ────────────────────────────────────
def _anthropic_raw(text=None, tool_use=None, stop_reason="end_turn",
                   input_tokens=10, output_tokens=5, model="claude-3-5-sonnet"):
    blocks = []
    if text is not None:
        b = MagicMock(); b.type = "text"; b.text = text; blocks.append(b)
    if tool_use is not None:
        b = MagicMock(); b.type = "tool_use"; b.id = "tu1"
        b.name = "search"; b.input = {"q": "ramen"}; blocks.append(b)
    raw = MagicMock()
    raw.content = blocks
    raw.stop_reason = stop_reason
    raw.model = model
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    raw.usage = usage
    return raw


def test_anthropic_transform_response_text():
    cfg = AnthropicConfig()
    raw = _anthropic_raw(text="hello world", stop_reason="end_turn")
    resp = cfg.transform_response(
        raw, model="claude", preset_id="p1", elapsed_ms=50.0,
        cost_per_1k_input_tokens=3.0, cost_per_1k_output_tokens=15.0,
    )
    assert isinstance(resp, ModelResponse)
    assert resp.choices[0].message.content == "hello world"
    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert resp.preset_id == "p1"
    # 10 * 3/1000 + 5 * 15/1000 = 0.03 + 0.075 = 0.105
    assert resp.cost_usd == pytest.approx(0.105)


def test_anthropic_transform_response_tool_use():
    cfg = AnthropicConfig()
    raw = _anthropic_raw(text=None, tool_use=True, stop_reason="tool_use")
    resp = cfg.transform_response(
        raw, model="claude", preset_id="p1", elapsed_ms=0,
        cost_per_1k_input_tokens=0, cost_per_1k_output_tokens=0,
    )
    assert resp.choices[0].finish_reason == "tool_calls"
    tc = resp.choices[0].message.tool_calls
    assert len(tc) == 1
    assert tc[0]["id"] == "tu1"
    assert tc[0]["type"] == "function"
    assert tc[0]["function"]["name"] == "search"
    assert '"q": "ramen"' in tc[0]["function"]["arguments"]


def test_anthropic_transform_response_max_tokens_finish():
    cfg = AnthropicConfig()
    raw = _anthropic_raw(text="cut off", stop_reason="max_tokens")
    resp = cfg.transform_response(
        raw, model="claude", preset_id="p1", elapsed_ms=0,
        cost_per_1k_input_tokens=0, cost_per_1k_output_tokens=0,
    )
    assert resp.choices[0].finish_reason == "length"


# ── AnthropicConfig: make_client lazy import ───────────────────────────────
def test_anthropic_make_client_raises_without_anthropic_sdk(monkeypatch):
    """If anthropic SDK is not installed, selecting an anthropic preset
    raises a clear ImportError (not at module import time)."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("simulated: not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = AnthropicConfig()
    with pytest.raises(ImportError, match="anthropic"):
        cfg.make_client("https://api.anthropic.com", "sk-test")


# ── PresetConfig ────────────────────────────────────────────────────────────
def test_preset_resolve_api_key_from_env(monkeypatch):
    p = PresetConfig(
        id="p", label="P", base_url="x", api_key_env="MY_KEY", adapter="openai_compat",
    )
    monkeypatch.setenv("MY_KEY", "sk-123")
    assert p.resolve_api_key() == "sk-123"


def test_preset_resolve_api_key_explicit_override_env(monkeypatch):
    p = PresetConfig(
        id="p", label="P", base_url="x", api_key_env="MY_KEY",
        adapter="openai_compat", api_key="sk-explicit",
    )
    monkeypatch.setenv("MY_KEY", "sk-from-env")
    # explicit api_key takes precedence
    assert p.resolve_api_key() == "sk-explicit"


def test_preset_model_for_role():
    p = PresetConfig(
        id="p", label="P", base_url="x", api_key_env="K", adapter="openai_compat",
        models={"generator": "g1", "evaluator": "e1"},
    )
    assert p.model_for("generator") == "g1"
    assert p.model_for("evaluator") == "e1"
    assert p.model_for("missing") == ""


def test_preset_extra_body_for_thinking():
    p = PresetConfig(
        id="p", label="P", base_url="x", api_key_env="K", adapter="openai_compat",
        thinking_extra_body={"disabled": {"thinking": {"type": "disabled"}}},
    )
    assert p.extra_body_for(thinking=False) == {"thinking": {"type": "disabled"}}
    # this endpoint declares no way to force thinking on
    assert p.extra_body_for(thinking=True) is None
    # no preference from the caller -> leave the endpoint's default alone
    assert p.extra_body_for(thinking=None) is None
