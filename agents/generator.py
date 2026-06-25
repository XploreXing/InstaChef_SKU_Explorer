import json
import os
from openai import OpenAI
from utils.generator_tools import (
    GENERATOR_TOOL_SCHEMAS, build_handlers
)
from utils.json_parser import parse_llm_json
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
    #search_summary: str,
    feedback: str,
    locked_names: list[str],
    round_num: int,
) -> str:
    #parts = [search_summary, ""]
    parts=[] #不要search_summary了，原因是什么待查
    #parts用来构建generator的user message？

    if feedback:
        parts.append("## IMPROVEMENT FEEDBACK FROM PREVIOUS ROUND")
        parts.append(feedback)
        parts.append("")
        parts.append("## CURRENT STATUS")
        parts.append(f"- {len(locked_names)} proposals already accepted for {cuisine}")
        parts.append(f"- This is round {round_num} of 3")
        parts.append("")
        parts.append("Do NOT repeat these accepted proposals:")
        parts.append(json.dumps(locked_names, indent=2, ensure_ascii=False))
    else:
        parts.append("## FIRST ROUND — no prior feedback. Generate fresh proposals.")
        parts.append("Use discover_cuisine to research market trends, then generate fresh proposals.")
    parts.append("")
    parts.append(f"Generate {count} dish proposals for {cuisine} cuisine.")
    return "\n".join(parts)


class GeneratorAgent:
    def __init__(self, config: dict):
        self.config=config
        self.cfg = config["llm"]
        api_key = self.cfg.get("api_key") or os.getenv(self.cfg.get("api_key_env", ""))
        self.client = OpenAI(
            base_url=self.cfg.get("base_url", "https://api.siliconflow.cn/v1"),
            api_key=api_key,
        )
        self.system_prompt = _load_system_prompt()
        self.tool_context=ToolContext() #每个实例一个，线程安全？
    def generate(
        self,
        cuisine: str,
        count: int,
        #search_summary: str,
        feedback: str = "",
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
            #search_summary=search_summary,
            feedback=feedback,
            locked_names=locked_names,
            round_num=round_num,
        )
        messages=[{"role":"system","content":self.system_prompt},
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

    def _run_tool_loop(self, messages, handlers, ctx) -> list[dict] | None:
        """Stage 1: standard tool-using agent loop.

        Returns parsed proposals on success, None if the loop exhausted without
        a parseable JSON answer (so the caller can fall back).
        """
        for loop_i in range(10):
            response = self.client.chat.completions.create(
                model=self.cfg["generator_model"],
                messages=messages,
                temperature=self.cfg["generator_temperature"],
                max_tokens=4096,
                tools=GENERATOR_TOOL_SCHEMAS,
            )
            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            messages.append(msg)

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
                response = self.client.chat.completions.create(
                    model=self.cfg["generator_model"],
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
