import json
import os
from openai import BadRequestError
from utils.generator_tools import (
    GENERATOR_TOOL_SCHEMAS, build_handlers
)
from utils.json_parser import parse_llm_json
from utils.llm_providers.base import is_prompt_too_long_error  # noqa: F401 (re-exported for test compat)
from utils.llm_router import LLMRouter
from models import ToolContext
GENERATOR_SYSTEM_PROMPT = None


def _load_system_prompt() -> str:
    global GENERATOR_SYSTEM_PROMPT
    if GENERATOR_SYSTEM_PROMPT is None:
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "generator_system.md"
        )
        with open(prompt_path, "r") as f:
            GENERATOR_SYSTEM_PROMPT = f.read()
    return GENERATOR_SYSTEM_PROMPT


def build_generator_user_message(
    cuisine: str,
    count: int,
    locked_names: list[str],
    feedback: str = "",
    round_num: int = 1,
) -> str:
    parts = []

    if round_num > 1:
        parts.append("## CURRENT STATUS")
        parts.append(f"- {len(locked_names)} proposals already accepted for {cuisine}")
        parts.append(f"- This is round {round_num} of 3")
        parts.append("")
        if feedback:
            parts.append(f"The feedback from evaluator last round is: {feedback}")
        parts.append("Do NOT repeat these accepted proposals:")
        parts.append(json.dumps(locked_names, indent=2, ensure_ascii=False))
    else:
        parts.append("## FIRST ROUND — no prior feedback. Generate fresh proposals.")
        parts.append("Use discover_cuisine to research market trends, then generate fresh proposals.")
    parts.append("")
    parts.append(f"Generate {count} dish proposals for {cuisine} cuisine.")
    return "\n".join(parts)


class GeneratorAgent:
    def __init__(self, config: dict, breaker=None):
        self.config=config
        self.cfg = config["llm"]
        self.client = LLMRouter(
            self.cfg,
            role="generator",
            selected_preset_id=self.cfg.get("selected_preset_id"),
            breaker=breaker,
        )
        self.system_prompt = _load_system_prompt()
        self.tool_context=ToolContext() #每个实例一个，线程安全？
    def generate(
        self,
        cuisine: str,
        count: int,
        #search_summary: str,
        feedback: str = "",
        HITL_feedback:str="",
        existing_skus=None,
        locked_names: list[str] | None = None,
        round_num: int = 1,
    ) -> list[dict]:
        if locked_names is None:
            locked_names = []
        #在这个agent开始 generation call之前 创建新的ToolContext
        ctx=ToolContext(
            cuisine=cuisine,
            existing_skus=existing_skus or [], #提问：这个existing_skus在哪一步读取，是放在内存中吗？
            config=self.config, #为什么这里要传入完整的配置文件？我猜是为了读取设定好的query templates
            breaker=self.client.breaker,
        )

        # Preserve source_urls from previous rounds (discover_cuisine should
        # only be called once; this lets the Agent reuse earlier research)
        if self.tool_context.source_urls:
            ctx.source_urls=dict(self.tool_context.source_urls)
            ctx.source_contents=dict(self.tool_context.source_contents) #又忘记了ToolContext是每个菜系的实例前面搜索tavily搜索的内容集合？？
        self.tool_context=ctx #如果前一轮有source_urls则有if那一步添加好了之后再更新？这一步没理解，否则的话直接用前面初始化的ctx?

        user_message = build_generator_user_message(
            cuisine=cuisine,
            count=count,
            locked_names=locked_names,
            feedback=feedback,
            round_num=round_num,
        )
        system_content = self.system_prompt
        if HITL_feedback:
            system_content += (
                "\n\n## EXPERT FEEDBACK CONSTRAINTS (from human review)\n"
                "The following feedback comes from human experts who reviewed "
                "and REJECTED previous proposals. Treat these as hard "
                "constraints — do NOT repeat these mistakes:\n"
                + HITL_feedback
            )
        messages=[{"role":"system","content":system_content},
                  {"role":"user","content":user_message}]
        handlers=build_handlers(ctx) #传入了以后，这些handlers没有调用，但是都有了上下文
        try:
            # Stage 1: normal tool-use loop (discover research + halal/dup checks)
            proposals = self._run_tool_loop(messages, handlers, ctx)
            if proposals:
                return proposals

            # Stage 2: recover JSON that the model may have emitted alongside a
            # tool_call in an earlier assistant turn (case that `continue` skipped)
            proposals = self._recover_from_history(messages, ctx)
            if proposals:
                return proposals

            # Stage 3: model finished its tool research but never emitted valid
            # JSON (e.g. returned markdown prose). Reuse the full conversation
            # history — research is already captured — but drop tools and force
            # a pure-JSON final answer.
            print(f"[generator] {cuisine} r{round_num}: tool loop empty, "
                  f"forcing no-tools JSON fallback", flush=True)
            proposals = self._force_final_output(messages, ctx, kind="proposals")
            return proposals or []

        except Exception as e:
            print(f"Generator API call failed: {e}", flush=True)
            return []

    def _attach_source_urls(self, proposals: list[dict], ctx) -> list[dict]:
        """Resolve REF_XX tags in each proposal to concrete URLs from ctx."""
        for p in proposals:
            ref_tags = p.pop("source_refs", [])
            p["source_urls"] = [
                ctx.source_urls[tag]
                for tag in ref_tags if tag in ctx.source_urls
            ]
        return proposals
    def _compact_tool_messages(self, messages):
        """结构化截断：只改 tool message 的 content，保留 message 本身和
        tool_call_id（否则与上方 assistant tool_call 配对断裂 → API 400）。

        只压"偏大"的 tool 结果（小的 follow-up 查询结果保留不动），正文换成存根，
        REF 标签保留在存根里让模型仍知道有哪些源；URL 全程在 ctx 中，溯源不断。
        """
        import re
        # 阈值取当前所有 tool message content 的中位数，只压明显偏大的；若不足
        # 两条则保证至少压最大的那条。
        # history 里的 assistant 消息是 SDK 返回的对象（没有 .get）；tool 消息
        # 都是 _run_tool_loop 自己 append 的 dict，所以只在 dict 里找。
        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        sizes = [len(m.get("content", "")) for m in tool_msgs]
        threshold = sorted(sizes)[len(sizes) // 2] if sizes else 0
        tool_msgs.sort(key=lambda m: len(m.get("content", "")), reverse=True)
        n_compact = max(1, sum(1 for s in sizes if s > threshold and s > 0))
        for m in tool_msgs[:n_compact]:
            refs = re.findall(r"REF_\d{2}", m.get("content", ""))
            ref_str = ", ".join(sorted(set(refs))) or "n/a"
            m["content"] = (
                f"[omitted to save context — source URLs retained in ctx; "
                f"refs: {ref_str}]"
            )

    def _run_tool_loop(self, messages, handlers, ctx) -> list[dict] | None:
        """Stage 1: standard tool-using agent loop.

        Returns parsed proposals on success, None if the loop exhausted without
        a parseable JSON answer (so the caller can fall back).
        """
        has_compacted = False
        for loop_i in range(10):
            try:
                response = self.client.chat(
                    messages=messages,
                    temperature=self.cfg["generator_temperature"],
                    max_tokens=4096,
                    tools=GENERATOR_TOOL_SCHEMAS,
                )
                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                messages.append(msg)
            except Exception as e:
                # Path 2: prompt too long -> reactive compact (once), then retry.
                # 第二次仍 too-long，或非 too-long 异常（401/529 等）一律 raise，
                # 交给顶层 generate() 的 except 分流——绝不 fall-through 到下面
                # 使用未定义的 msg（否则真实错误会被 NameError 掩盖）。
                if is_prompt_too_long_error(e) and not has_compacted:
                    self._compact_tool_messages(messages)
                    has_compacted = True
                    print(f"[generator] prompt too long, compacted oldest tool "
                          f"messages, retrying", flush=True)
                    continue
                raise

            # Case 1: Model wants to call a tool
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    handler = handlers.get(tc.function.name)
                    if handler:
                        args = json.loads(tc.function.arguments)
                        result = handler(args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                continue  # loop back, model can call more tools or output final

            # Case 2: Model outputs final response (JSON proposals)
            if msg.content:
                proposals = self._parse_response(msg.content)
                if proposals:
                    return self._attach_source_urls(proposals, ctx)
                print(f"[generator] loop {loop_i}: content returned but parse failed, "
                      f"content preview: {msg.content[:200]!r}", flush=True)
                continue

            # Case 3: No tool_calls, no content — model returned nothing
            print(f"[generator] loop {loop_i}: empty content, no tool_calls, "
                  f"finish_reason={finish_reason}", flush=True)
            if finish_reason in ("stop", "content_filter"):
                # stop/content_filter: model said nothing — give Stage 2/3 a chance
                break
            # other reasons (length, etc.) — try another loop iteration

        return None

    def _recover_from_history(self, messages, ctx) -> list[dict] | None:
        """Stage 2: salvage JSON emitted in an earlier assistant turn.

        Covers the case where the model produced its final content in the SAME
        message as a tool_call — Stage 1's `continue` would have skipped it.
        """
        for msg in reversed(messages):
            if getattr(msg, "role", None) != "assistant":
                continue
            content = getattr(msg, "content", None)
            if not content:
                continue
            proposals = self._parse_response(content)
            if proposals:
                print(f"[generator] recovered proposals from earlier assistant turn",
                      flush=True)
                return self._attach_source_urls(proposals, ctx)
        return None

    def _force_final_output(self, messages, ctx, kind: str = "proposals") -> list[dict] | None:
        """Stage 3: drop tools, force a pure-JSON final answer.

        The conversation history (incl. discover_cuisine research kept as tool
        messages) is reused, so prior work is not wasted — the model just has to
        re-format it as JSON without the option to call more tools.
        """
        nudge = {
            "proposals": ("Output ONLY a JSON object with a top-level \"proposals\" "
                          "key containing the dish proposals. No markdown (no ###, "
                          "no **, no -), no prose, no explanation. "
                          "Start with { and end with }."),
        }[kind]
        forced_messages = list(messages) + [{"role": "user", "content": nudge}]

        for attempt in range(2):
            try:
                response = self.client.chat(
                    messages=forced_messages,
                    temperature=0,            # deterministic final formatting
                    max_tokens=4096,
                    # no tools= here -> model cannot stall on tool calls
                    # response_format may be ignored by some providers; we don't
                    # rely on it, the prompt nudge is the real enforcer.
                    response_format={"type": "json_object"},
                )
                msg = response.choices[0].message
                fr = response.choices[0].finish_reason

                if msg.content:
                    proposals = self._parse_response(msg.content)
                    if proposals:
                        print(f"[generator] forced output succeeded on attempt {attempt}",
                              flush=True)
                        return self._attach_source_urls(proposals, ctx)
                    print(f"[generator] forced attempt {attempt}: parse fail, "
                          f"content preview: {msg.content[:200]!r}", flush=True)
                    forced_messages.append(msg)
                    forced_messages.append({"role": "user",
                        "content": "That was not valid JSON. Output ONLY the JSON object now."})
                else:
                    print(f"[generator] forced attempt {attempt}: empty content, "
                          f"finish_reason={fr}", flush=True)
            except Exception as e:
                print(f"[generator] forced attempt {attempt} error: {e}", flush=True)
        print(f"[generator] forced-output exhausted, returning []", flush=True)
        return None

    def _parse_response(self, content: str) -> list[dict]:
        result = parse_llm_json(content, top_key="proposals")
        if result is None:
            # No noisy log here — the loop / fallback stages log context instead.
            return []
        if isinstance(result, list):
            return result
        return []
