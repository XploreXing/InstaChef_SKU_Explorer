import json
import os
import traceback
from utils.llm_router import LLMRouter
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


# Weights of the two dish-level dimensions, kept in the ratio they had in the
# original rubric (35 : 25). The total is computed here, never by the model.
DISH_WEIGHTS = {"trend_heat": 3.5, "hawker_substitutability": 2.5}


def dish_score(trend_heat: float, hawker_substitutability: float) -> float:
    """A dish's score on 0-100 from its two dimension scores (0-10 each)."""
    points = (trend_heat * DISH_WEIGHTS["trend_heat"]
              + hawker_substitutability * DISH_WEIGHTS["hawker_substitutability"])
    return round(points / (10 * sum(DISH_WEIGHTS.values())) * 100, 1)


def _raw_score(scores: dict, dimension: str) -> float:
    """The model's 0-10 score for one dimension, or 0 if it is missing or not a number."""
    entry = scores.get(dimension)
    raw = entry.get("raw") if isinstance(entry, dict) else entry
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    return min(max(float(raw), 0.0), 10.0)


# Veto reason given to a proposal the evaluator returned no evaluation for.
NOT_SCORED_REASON = "Evaluator 未返回该菜的评分"


def _match_key(name) -> str:
    return " ".join(str(name or "").lower().split())


def build_evaluator_user_message(proposals: list[dict], hitl_context: str = "") -> str:
    """The evaluator sees the dishes and nothing about the run: how many are
    wanted or already found is not its business, only how good each dish is."""
    parts = [
        "## Proposals to Evaluate (all hard-constraint checks already passed)",
        json.dumps(proposals, indent=2, ensure_ascii=False),
    ]

    if hitl_context:
        parts.append(hitl_context)

    parts.extend([
        "",
        "Score every proposal on the 2 dimensions.",
    ])
    return "\n".join(parts)


class EvaluatorAgent:
    def __init__(self, config: dict, breaker=None):
        self.cfg = config["llm"]
        self.client = LLMRouter(
            self.cfg,
            role="evaluator",
            selected_preset_id=self.cfg.get("selected_preset_id"),
            breaker=breaker,
        )
        self.system_prompt = _load_system_prompt()

    def evaluate(
        self,
        proposals: list[dict],
        hitl_context: str = "",
    ) -> tuple[list[dict], dict, str]:

        user_message = build_evaluator_user_message(proposals, hitl_context)

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ]
            handlers = {
                "search_web_for_eval": lambda args: search_web_for_eval(**args),
            }

            # Stage 1: normal tool-use loop
            try:
                result = self._run_tool_loop(messages, handlers)
            except Exception as e:
                # The model may have answered before the failure, and re-reading
                # history costs no API call. Stage 3 is skipped: it re-sends the
                # same history, so it would only repeat an API failure.
                print(f"[evaluator] stage 1 raised "
                      f"{type(e).__name__}: {e}; salvaging from history", flush=True)
                traceback.print_exc()
                return self._recover_from_history(messages) or ([], {}, "")
            if result:
                return result

            # Stage 2: salvage JSON emitted alongside an earlier tool_call
            result = self._recover_from_history(messages)
            if result:
                return result

            # Stage 3: model finished research without emitting valid JSON
            print(f"[evaluator] tool loop empty, "
                  f"forcing no-tools JSON fallback", flush=True)
            result = self._force_final_output(messages)
            return result or ([], {}, "")
        except Exception as e:
            print(f"Evaluator API call failed: {e}")
            traceback.print_exc()
            return [], {}, ""

    def _run_tool_loop(self, messages, handlers) -> tuple[list[dict], dict, str] | None:
        """Stage 1: standard tool-using loop. Returns parsed tuple or None."""
        for loop_i in range(5):
            response = self.client.chat(
                messages=messages,
                temperature=self.cfg["evaluator_temperature"],
                max_tokens=8192,
                tools=EVALUATOR_TOOL_SCHEMAS,
                thinking=True,  # scoring is where reasoning before answering pays off
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
                    # summary is an object per the output schema; str() before slicing
                    print(f"[evaluator] loop {loop_i}: content parsed successful"
                          f"[summary]:{str(summary)[:500]}"
                          f"[suggestion]: {str(suggestions)[:500]}"
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
                response = self.client.chat(
                    messages=forced_messages,
                    temperature=0,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                    thinking=True,
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
        """One result per proposal, in proposal order.

        An evaluation is matched to its proposal by dish name, then by id: the
        model may reorder or drop dishes, so position is not trusted. The total
        is computed here from the dimension scores, and a dish stays a
        candidate (`passed`) unless it was vetoed. Whatever total or verdict
        the model reported is ignored.
        """
        evaluations = [ev for ev in evaluations if isinstance(ev, dict)]
        by_name = {_match_key(ev.get("name")): ev for ev in evaluations}
        by_id = {ev.get("id"): ev for ev in evaluations}
        proposal_ids = [p.get("id") for p in proposals]

        results = []
        for prop in proposals:
            proposal = DishProposal(
                id=prop.get("id", 0),
                name=prop.get("name", ""),
                name_cn=prop.get("name_cn", ""),
                cuisine=prop.get("cuisine", ""),
                price_sgd=prop.get("price_sgd", 0.0),
                description=prop.get("description", ""),
                description_cn=prop.get("description_cn", ""),
                differentiation=prop.get("differentiation", ""),
                trend_source=prop.get("trend_source", ""),
                source_urls=prop.get("source_urls", prop.get("source_refs", [])),
            )

            ev = by_name.get(_match_key(prop.get("name")))
            if ev is None and prop.get("id") is not None and proposal_ids.count(prop.get("id")) == 1:
                ev = by_id.get(prop.get("id"))
            if ev is None:
                # Kept visible, but out of the ranking: there is nothing to rank it by.
                results.append(EvaluationResult(
                    proposal=proposal,
                    vetoed=True,
                    veto_reason=NOT_SCORED_REASON,
                    trend_heat=0,
                    hawker_substitutability=0,
                    total_score=0,
                    passed=False,
                    reasoning="",
                ))
                continue

            scores = ev.get("scores") if isinstance(ev.get("scores"), dict) else {}
            vetoed = bool(ev.get("vetoed", False))
            trend_heat = 0.0 if vetoed else _raw_score(scores, "trend_heat")
            hawker = 0.0 if vetoed else _raw_score(scores, "hawker_substitutability")
            results.append(EvaluationResult(
                proposal=proposal,
                vetoed=vetoed,
                veto_reason=(ev.get("veto_reason") or "Evaluator 否决（未给出理由）") if vetoed else None,
                trend_heat=trend_heat,
                hawker_substitutability=hawker,
                total_score=dish_score(trend_heat, hawker),
                passed=not vetoed,
                reasoning=json.dumps(scores, ensure_ascii=False),
            ))
        return results
