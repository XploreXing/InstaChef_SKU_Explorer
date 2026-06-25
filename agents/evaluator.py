import json
import os
from openai import OpenAI
from models import DishProposal, EvaluationResult
from utils.evaluator_tools import (search_web_for_eval, EVALUATOR_TOOL_SCHEMAS)
from utils.json_parser import parse_llm_json

EVALUATOR_SYSTEM_PROMPT = None


def _load_system_prompt() -> str:
    global EVALUATOR_SYSTEM_PROMPT
    if EVALUATOR_SYSTEM_PROMPT is None:
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "evaluator_system.md"
        )
        with open(prompt_path, "r") as f:
            EVALUATOR_SYSTEM_PROMPT = f.read()
    return EVALUATOR_SYSTEM_PROMPT


def _serialize_commodities(commodities) -> list[dict]:
    """Serialize commodities for Evaluator prompt.
    Keeps cuisine_type so blue-ocean scoring knows the count per cuisine."""
    return [
        {"name": c.name, "cuisine_type": c.cuisine_type, "description": c.description[:120]}
        for c in commodities
    ]


def build_evaluator_user_message(
    proposals: list[dict],
    cuisine_sku_count: int,
    round_num: int,
    locked_count: int,
    remaining: int,
    pass_threshold: int = 80,
    hitl_context: str = "",
) -> str:
    parts = [
        f"## Cuisine Context",
        f"Current SKU count for this cuisine: {cuisine_sku_count}",
        f"(Fewer SKUs → higher Blue Ocean score)",
        "",
        "## Proposals to Evaluate (all hard-constraint checks already passed)",
        json.dumps(proposals, indent=2, ensure_ascii=False),
        "",
        "## Progress",
        f"Round {round_num} of 3.",
        f"Currently {locked_count} accepted, need {remaining} more.",
        f"Pass threshold: total_score >= {pass_threshold}.",
    ]

    if hitl_context:
        parts.append(hitl_context)

    parts.extend([
        "",
        "Evaluate each proposal against the 4 hard constraints first.",
        "If any constraint fails, mark as vetoed with score 0.",
        "Only score the 3 dimensions if ALL constraints pass.",
    ])
    return "\n".join(parts)


class EvaluatorAgent:
    def __init__(self, config: dict):
        self.cfg = config["llm"]
        api_key = self.cfg.get("api_key") or os.getenv(self.cfg.get("api_key_env", ""))
        self.client = OpenAI(
            base_url=self.cfg.get("base_url", "https://api.siliconflow.cn/v1"),
            api_key=api_key,
        )
        self.system_prompt = _load_system_prompt()

    def evaluate(
        self,
        proposals: list[dict],
        cuisine_sku_count: int,
        round_num: int = 1,
        locked_count: int = 0,
        remaining: int = 10,
        pass_threshold: int = 80,
        hitl_context: str = "",
    ) -> tuple[list[dict], dict, str]:
        
        user_message = build_evaluator_user_message(
            proposals=proposals,
            cuisine_sku_count=cuisine_sku_count,
            round_num=round_num, 
            locked_count=locked_count,
            remaining=remaining,
            pass_threshold=pass_threshold,
            hitl_context=hitl_context,
        )
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ]
            handlers = {
                "search_web_for_eval": lambda args: search_web_for_eval(**args),
            }

            # Stage 1: normal tool-use loop
            result = self._run_tool_loop(messages, handlers)
            if result:
                return result

            # Stage 2: salvage JSON emitted alongside an earlier tool_call
            result = self._recover_from_history(messages)
            if result:
                return result

            # Stage 3: model finished research without emitting valid JSON
            print(f"[evaluator] r{round_num}: tool loop empty, "
                  f"forcing no-tools JSON fallback", flush=True)
            result = self._force_final_output(messages)
            return result or ([], {}, "")
        except Exception as e:
            print(f"Evaluator API call failed: {e}")
            return [], {}, ""

    def _run_tool_loop(self, messages, handlers) -> tuple[list[dict], dict, str] | None:
        """Stage 1: standard tool-using loop. Returns parsed tuple or None."""
        for loop_i in range(5):
            response = self.client.chat.completions.create(
                model=self.cfg["evaluator_model"],
                messages=messages,
                temperature=self.cfg["evaluator_temperature"],
                max_tokens=8192,
                tools=EVALUATOR_TOOL_SCHEMAS,
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
                continue  # loop back, let model process tool results

            # Case 2: Model outputs final JSON response
            if msg.content:
                evaluations, summary, suggestions = self._parse_response(msg.content)
                if evaluations:
                    print(f"[evaluator] loop {loop_i}: content parsed successful" 
                          f"[summary]:{summary[:500]}"
                          f"[suggestion]: {suggestions[:500]}"
                        f"content preview: {msg.content[:200]!r}", flush=True)
                    return evaluations, summary, suggestions
                print(f"[evaluator] loop {loop_i}: content returned but parse failed, "
                      f"content preview: {msg.content[:200]!r}", flush=True)
                continue

            # Case 3: No tool_calls, no content
            print(f"[evaluator] loop {loop_i}: empty content, no tool_calls, "
                  f"finish_reason={finish_reason}", flush=True)
            if finish_reason in ("stop", "content_filter"):
                break
        return None

    def _recover_from_history(self, messages) -> tuple[list[dict], dict, str] | None:
        """Stage 2: salvage JSON emitted in an earlier assistant turn."""
        for msg in reversed(messages):
            if getattr(msg, "role", None) != "assistant":
                continue
            content = getattr(msg, "content", None)
            if not content:
                continue
            evaluations, summary, suggestions = self._parse_response(content)
            if evaluations:
                print(f"[evaluator] recovered evaluations from earlier assistant turn",
                      flush=True)
                return evaluations, summary, suggestions
        return None

    def _force_final_output(self, messages) -> tuple[list[dict], dict, str] | None:
        """Stage 3: drop tools, force a pure-JSON final evaluation answer."""
        nudge = ("Output ONLY a JSON object with top-level keys \"evaluations\", "
                 "\"summary\", and \"improvement_suggestions\". No markdown "
                 "(no ###, no **, no -), no prose, no explanation. "
                 "Start with { and end with }.")
        forced_messages = list(messages) + [{"role": "user", "content": nudge}]

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg["evaluator_model"],
                    messages=forced_messages,
                    temperature=0,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )
                msg = response.choices[0].message
                fr = response.choices[0].finish_reason

                if msg.content:
                    evaluations, summary, suggestions = self._parse_response(msg.content)
                    if evaluations:
                        print(f"[evaluator] forced output succeeded on attempt {attempt}",
                              flush=True)
                        return evaluations, summary, suggestions
                    print(f"[evaluator] forced attempt {attempt}: parse fail, "
                          f"content preview: {msg.content[:200]!r}", flush=True)
                    forced_messages.append(msg)
                    forced_messages.append({"role": "user",
                        "content": "That was not valid JSON. Output ONLY the JSON object now."})
                else:
                    print(f"[evaluator] forced attempt {attempt}: empty content, "
                          f"finish_reason={fr}", flush=True)
            except Exception as e:
                print(f"[evaluator] forced attempt {attempt} error: {e}", flush=True)
        print(f"[evaluator] forced-output exhausted, returning empty", flush=True)
        return None

    def _parse_response(self, content: str) -> tuple[list[dict], dict, str]:
        data = parse_llm_json(content)
        if data is None or not isinstance(data, dict):
            return [], {}, ""
        evaluations = data.get("evaluations", [])
        summary = data.get("summary", {})
        suggestions = data.get("improvement_suggestions", "")
        return evaluations, summary, suggestions

    @staticmethod
    def to_evaluation_results(
        evaluations: list[dict],
        proposals: list[dict],
    ) -> list[EvaluationResult]:
        results = []
        for i, ev in enumerate(evaluations):
            prop_dict = proposals[i] if i < len(proposals) else {
                "name": ev.get("name", "Unknown"),
                "cuisine": ev.get("cuisine", "Unknown"),
                "price_sgd": 0.0,
                "description": "",
                "differentiation": "",
                "trend_source": "",
            }
            proposal = DishProposal(
                id=ev.get("id", i + 1),
                name=ev.get("name", prop_dict.get("name", "")),
                name_cn=prop_dict.get("name_cn", ev.get("name_cn", "")),
                cuisine=ev.get("cuisine", prop_dict.get("cuisine", "")),
                price_sgd=prop_dict.get("price_sgd", 0.0),
                description=prop_dict.get("description", ""),
                description_cn=prop_dict.get("description_cn", ev.get("description_cn", "")),
                differentiation=prop_dict.get("differentiation", ""),
                trend_source=prop_dict.get("trend_source", ""),
                source_urls=prop_dict.get("source_urls", prop_dict.get("source_refs", [])),
            )

            scores = ev.get("scores", {})
            result = EvaluationResult(
                proposal=proposal,
                vetoed=ev.get("vetoed", False),
                veto_reason=(
                    ev.get("summary", {}).get("rejection_reasons", [{}])[0].get("reason")
                    if ev.get("vetoed") else None
                ),
                cuisine_blue_ocean=scores.get("cuisine_blue_ocean", {}).get("raw", 0),
                trend_heat=scores.get("trend_heat", {}).get("raw", 0),
                hawker_substitutability=scores.get("hawker_substitutability", {}).get("raw", 0),
                total_score=ev.get("total_score", 0),
                passed=ev.get("passed", False),
                reasoning=json.dumps(scores, ensure_ascii=False),
            )
            results.append(result)
        return results
