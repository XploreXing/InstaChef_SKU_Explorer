"""测试 Generator 的 prompt-too-long 响应式压缩恢复：

1. is_prompt_too_long_error 的判定边界（400 且措辞命中 → True；401/529 → False）
2. _compact_tool_messages 只改 content、保留 tool_call_id 配对不破坏
3. _run_tool_loop 在 prompt-too-long 时压缩一次并重试成功
4. _run_tool_loop 对非 too-long 异常直接抛出（不吞、不死循环、不被 NameError 掩盖）
"""
import json
import pytest
from openai import BadRequestError, AuthenticationError, RateLimitError
from unittest.mock import MagicMock

from agents.generator import (
    GeneratorAgent,
    is_prompt_too_long_error,
)


def _make_agent():
    """构造一个不连真实 API 的 GeneratorAgent（client 随后 monkeypatch）。"""
    cfg = {
        "llm": {
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key_env": "LLM_API_KEY",
            "generator_model": "deepseek-ai/DeepSeek-V3",
            "generator_temperature": 0.7,
        },
        "search": {"max_results_per_query": 5},
    }
    import os
    os.environ.setdefault("LLM_API_KEY", "test-key")
    return GeneratorAgent(cfg)


def _bad_request(body: str):
    """构造一个 BadRequestError（HTTP 400），把 message 文案塞进去。"""
    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {}
    return BadRequestError(body, response=resp, body=body)


# ── is_prompt_too_long_error ──

def test_too_long_detected_on_context_length_phrase():
    e = _bad_request(
        "This model's maximum context length is 8192 tokens. "
        "However, your messages resulted in 12345 tokens."
    )
    assert is_prompt_too_long_error(e) is True


def test_too_long_detected_on_prompt_is_too_long():
    e = _bad_request("Error: prompt is too long")
    assert is_prompt_too_long_error(e) is True


def test_401_not_mistaken_as_too_long():
    # AuthenticationError 不是 BadRequestError，必须立刻排除——绝不能误判后 compact。
    resp = MagicMock()
    resp.status_code = 401
    resp.headers = {}
    e = AuthenticationError("invalid api key", response=resp, body=None)
    assert is_prompt_too_long_error(e) is False


def test_429_not_mistaken_as_too_long():
    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {"retry-after": "1"}
    e = RateLimitError("rate limit exceeded", response=resp, body=None)
    assert is_prompt_too_long_error(e) is False


def test_other_400_not_mistaken_as_too_long():
    # 一个不含 context 措辞的 400（比如参数错误），不应被当成 too-long。
    e = _bad_request("Error: 'messages' is a required property")
    assert is_prompt_too_long_error(e) is False


# ── _compact_tool_messages ──

def test_compact_preserves_tool_call_id_and_shrinks_content():
    agent = _make_agent()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "discover_cuisine", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1",
         "content": "[REF_01] Trend A\n" + "x" * 5000},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_2", "function": {"name": "search_web", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_2",
         "content": "[REF_02] small result"},
    ]
    before_total = sum(len(m.get("content") or "") for m in messages)
    agent._compact_tool_messages(messages)
    after_total = sum(len(m.get("content") or "") for m in messages)

    # 体积必须显著下降
    assert after_total < before_total / 2
    # tool 消息本体仍在（只改了 content），tool_call_id 没丢——否则与上方 assistant
    # tool_call 配对断裂，API 会直接 400。
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"call_1", "call_2"}
    # 最大的那条被压成存根；小的不动
    big = next(m for m in tool_msgs if m["tool_call_id"] == "call_1")
    assert "omitted to save context" in big["content"]
    assert "REF_01" in big["content"]  # 存根里保留 REF，模型仍知道有哪些源
    small = next(m for m in tool_msgs if m["tool_call_id"] == "call_2")
    assert small["content"] == "[REF_02] small result"  # 未被改动


def test_compact_no_tool_messages_is_noop():
    agent = _make_agent()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    agent._compact_tool_messages(messages)  # 不应抛错
    assert len(messages) == 2


# ── _run_tool_loop reactive compact + retry ──

def _fake_create_side_effect(call_seq):
    """call_seq: list of either dict(responses) or Exception instances."""
    calls = {"n": 0}
    def _create(**kwargs):
        i = calls["n"]
        calls["n"] += 1
        item = call_seq[i]
        if isinstance(item, Exception):
            raise item
        return item
    return _create, calls


def _assistant_msg(content=None, tool_calls=None):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls
    return m


def _tool_call(cid, name="discover_cuisine"):
    tc = MagicMock()
    tc.id = cid
    tc.function.name = name
    tc.function.arguments = "{}"
    return tc


def _response(msg, finish_reason="stop"):
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message = msg
    r.choices[0].finish_reason = finish_reason
    return r


def test_run_tool_loop_compacts_then_retries_on_prompt_too_long(monkeypatch):
    """prompt-too-long -> 压缩 -> 重试成功。验证恢复路径打通、且只压一次。"""
    agent = _make_agent()

    # 第一次 create 抛 too-long；第二次正常返回 JSON proposals。
    ok_response = _response(
        _assistant_msg(content=json.dumps({"proposals": [{"name": "Dish X", "source_refs": []}]}))
    )
    seq = [_bad_request("This model's maximum context length is 8192 tokens."), ok_response]
    _create, calls = _fake_create_side_effect(seq)
    monkeypatch.setattr(agent.client, "chat", _create)

    # build_handlers 需要 ctx；给一个空 handlers 映射即可（第二次不调工具）
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "function": {"name": "discover_cuisine", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "x" * 8000},
        {"role": "user", "content": "go"},
    ]
    from utils.generator_tools import build_handlers
    from models import ToolContext
    ctx = ToolContext(cuisine="Test", existing_skus=[], config=agent.config)
    handlers = build_handlers(ctx)

    proposals = agent._run_tool_loop(messages, handlers, ctx)
    assert calls["n"] == 2  # 第一次失败、压缩后第二次成功
    assert proposals and proposals[0]["name"] == "Dish X"
    # 压缩后那条 tool 消息的 content 应已被改成存根
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert any("omitted to save context" in (m.get("content") or "") for m in tool_msgs)


def test_run_tool_loop_raises_on_non_toolong_error(monkeypatch):
    """非 too-long 异常（此处 401）必须直接抛出，不能被吞或变 NameError。"""
    agent = _make_agent()
    resp = MagicMock()
    resp.status_code = 401
    resp.headers = {}
    auth_err = AuthenticationError("invalid api key", response=resp, body=None)
    _create, _ = _fake_create_side_effect([auth_err])
    monkeypatch.setattr(agent.client, "chat", _create)

    from utils.generator_tools import build_handlers
    from models import ToolContext
    ctx = ToolContext(cuisine="Test", existing_skus=[], config=agent.config)
    handlers = build_handlers(ctx)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    with pytest.raises(AuthenticationError):
        agent._run_tool_loop(messages, handlers, ctx)


def test_run_tool_loop_second_too_long_raises(monkeypatch):
    """压缩一次后仍 too-long -> 不再压缩，直接抛出（避免无限重试）。"""
    agent = _make_agent()
    err = _bad_request("This model's maximum context length is 8192 tokens.")
    # 两次都抛 too-long
    _create, _ = _fake_create_side_effect([err, err])
    monkeypatch.setattr(agent.client, "chat", _create)

    from utils.generator_tools import build_handlers
    from models import ToolContext
    ctx = ToolContext(cuisine="Test", existing_skus=[], config=agent.config)
    handlers = build_handlers(ctx)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    with pytest.raises(BadRequestError):
        agent._run_tool_loop(messages, handlers, ctx)
