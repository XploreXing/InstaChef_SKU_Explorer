import time
import yaml
from models import (
    OrchestratorState, RoundResult, CuisineResult,
    FinalOutput, EvaluationResult, DishProposal, ProcessedCommodity,
)
from utils.data_loader import SKUDataLoader
from utils.guard import HardConstraintGuard
from agents.generator import GeneratorAgent
from agents.evaluator import EvaluatorAgent
from utils.search import FoodTrendSearcher


class Orchestrator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.state = OrchestratorState.IDLE
        self.results: dict[str, CuisineResult] = {}
        self.state_callbacks: dict[str, list] = {
            "on_state_change": [],
            "on_round_complete": [],
        }
        self.stop_check = None  # callable -> bool, set by caller to enable cancellation
        self.sku_loader: SKUDataLoader | None = None
        self.existing_skus: list[ProcessedCommodity] = []
        self.generator: GeneratorAgent | None = None
        self.evaluator: EvaluatorAgent | None = None
        self.searcher: FoodTrendSearcher | None = None

    def load_skus(self):
        self.sku_loader = SKUDataLoader(self.config)
        self.existing_skus = self.sku_loader.load()
        self.state = OrchestratorState.INIT

    def _init_agents(self):
        if self.generator is None:
            self.generator = GeneratorAgent(self.config)
        if self.evaluator is None:
            self.evaluator = EvaluatorAgent(self.config)
        if self.searcher is None:
            self.searcher = FoodTrendSearcher(self.config)

    def run(self, cuisines: list[str] | None = None) -> FinalOutput:
        targets = cuisines or self.config["orchestrator"]["cuisines"]
        self._init_agents()
        t_start = time.time()

        for cuisine in targets:
            if self.stop_check and self.stop_check():
                break
            result = self._process_cuisine(cuisine)
            self.results[cuisine] = result

        self.state = OrchestratorState.AGGREGATING
        output = FinalOutput(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            cuisines=self.results,
            total_elapsed_seconds=time.time() - t_start,
        )
        self.state = OrchestratorState.DONE
        return output

    def _process_cuisine(self, cuisine: str) -> CuisineResult:
        cfg = self.config["orchestrator"]
        locked: list[EvaluationResult] = []
        rounds_history: list[RoundResult] = []
        feedback = ""
        current_threshold = cfg["pass_threshold"]  # may auto-lower between rounds

        cuisine_skus = self.sku_loader.get_by_cuisine(cuisine)

        # Searching
        self._emit(cuisine, 0, "searching", {
            "locked_count": 0,
            "remaining": cfg["target_per_cuisine"],
        })
        search_results = self.searcher.search_cuisine(cuisine)

        # Summarizing
        self._emit(cuisine, 0, "summarizing", {
            "locked_count": 0,
            "remaining": cfg["target_per_cuisine"],
        })
        search_summary = self.searcher.summarize_for_generator(
            search_results, cuisine
        )

        for round_num in range(1, cfg["max_rounds_per_cuisine"] + 1):
            if self.stop_check and self.stop_check():
                break
            remaining = cfg["target_per_cuisine"] - len(locked)
            if remaining <= 0:
                break

            t_round = time.time()

            # GENERATING
            self.state = OrchestratorState.GENERATING
            self._emit(cuisine, round_num, "generating", {
                "locked_count": len(locked), "remaining": remaining
            })

            locked_names = [e.proposal.name for e in locked]
            proposal_dicts = self.generator.generate(
                cuisine=cuisine,
                count=remaining + 5,
                search_summary=search_summary,
                feedback=feedback,
                locked_names=locked_names,
                round_num=round_num,
            )

            if not proposal_dicts:
                break

            # --- Task 1: Hard-coded guard pre-check (before Evaluator LLM) ---
            guard_vetoed: list[EvaluationResult] = []
            guard_passed_proposals: list[dict] = []

            for p in proposal_dicts:
                passed, veto_reason = HardConstraintGuard.precheck(p)
                if not passed:
                    guard_vetoed.append(EvaluationResult(
                        proposal=DishProposal(
                            id=p.get("id", 0),
                            name=p.get("name", ""),
                            name_cn=p.get("name_cn", ""),
                            cuisine=p.get("cuisine", ""),
                            price_sgd=p.get("price_sgd", 0),
                            description=p.get("description", ""),
                            description_cn=p.get("description_cn", ""),
                            differentiation=p.get("differentiation", ""),
                            trend_source=p.get("trend_source", ""),
                        ),
                        vetoed=True,
                        veto_reason=veto_reason,
                        cuisine_blue_ocean=0,
                        trend_heat=0,
                        hawker_substitutability=0,
                        total_score=0,
                        passed=False,
                        reasoning="高压线熔断 — 无需 LLM 评估",
                    ))
                else:
                    guard_passed_proposals.append(p)

            # EVALUATING — only guard-passed proposals go to LLM
            self.state = OrchestratorState.EVALUATING
            self._emit(cuisine, round_num, "evaluating", {
                "proposal_count": len(proposal_dicts),
                "guard_vetoed": len(guard_vetoed),
                "locked_count": len(locked),
                "remaining": remaining,
            })

            # Only pass current cuisine's SKUs
            cuisine_skus = [
                s for s in self.existing_skus if s.cuisine_type == cuisine
            ]

            if guard_passed_proposals:
                evaluations_raw, summary, suggestions = self.evaluator.evaluate(
                    proposals=guard_passed_proposals,
                    existing_skus=cuisine_skus,
                    round_num=round_num,
                    locked_count=len(locked),
                    remaining=remaining,
                    pass_threshold=current_threshold,
                )

                if evaluations_raw:
                    llm_results = EvaluatorAgent.to_evaluation_results(
                        evaluations_raw, guard_passed_proposals
                    )
                else:
                    llm_results = []
            else:
                llm_results = []
                suggestions = ""

            # Combine: guard-vetoed + LLM-evaluated
            evaluation_results = guard_vetoed + llm_results

            # JUDGING — lock incrementally so UI sees step-by-step progress
            self.state = OrchestratorState.JUDGING
            new_passed = [e for e in evaluation_results if e.passed]
            new_rejected = [e for e in evaluation_results if not e.passed]

            for i, e in enumerate(new_passed):
                locked.append(e)
                display_locked = min(len(locked), cfg["target_per_cuisine"])
                self._emit(cuisine, round_num, "judging", {
                    "locked_count": display_locked,
                    "remaining": max(0, cfg["target_per_cuisine"] - display_locked),
                })
                if i < len(new_passed) - 1:
                    time.sleep(1)  # brief pause so UI poll catches each step

            round_result = RoundResult(
                cuisine=cuisine,
                round_num=round_num,
                proposals_generated=len(proposal_dicts),
                passed_count=len(new_passed),
                rejected_count=len(new_rejected),
                locked_total=len(locked),
                evaluations=evaluation_results,
                improvement_suggestions=suggestions,
                elapsed_seconds=time.time() - t_round,
            )
            rounds_history.append(round_result)
            self._emit_round_complete(round_result)

            # --- LLM semantic compression: synthesize feedback + decide threshold ---
            feedback, new_threshold = self._synthesize_feedback(
                evaluation_results=evaluation_results,
                cuisine=cuisine,
                pass_threshold=current_threshold,
                round_num=round_num,
                sku_count=len(cuisine_skus),
            )

            if new_threshold < current_threshold:
                print(f"🔧 Orchestrator: LLM lowered threshold {current_threshold}→{new_threshold}", flush=True)
                round_result.improvement_suggestions = (
                    f"[LLM auto-adjusted threshold {current_threshold}→{new_threshold}] "
                    + (round_result.improvement_suggestions or "")
                )
                current_threshold = new_threshold

            if current_threshold < cfg["pass_threshold"]:
                feedback += (
                    f"\n\nNote: pass threshold has been auto-adjusted to {current_threshold} "
                    f"(original: {cfg['pass_threshold']})."
                )

        return CuisineResult(
            cuisine=cuisine,
            total_rounds=len(rounds_history),
            locked=locked[:cfg["target_per_cuisine"]],
            rounds_history=rounds_history,
        )

    def _should_continue(self, locked_count: int, round_num: int) -> bool:
        cfg = self.config["orchestrator"]
        if locked_count >= cfg["target_per_cuisine"]:
            return False
        if round_num >= cfg["max_rounds_per_cuisine"]:
            return False
        return True

    def _synthesize_feedback(
        self,
        evaluation_results: list[EvaluationResult],
        cuisine: str,
        pass_threshold: int,
        round_num: int,
        sku_count: int,
    ) -> tuple[str, int]:
        """LLM-based semantic compression: summarize evaluator findings and
        produce actionable feedback for the Generator's next round.
        Returns (feedback_text, recommended_threshold)."""
        import json as _json
        import os as _os

        # Build a compact round summary for the LLM
        passed = [e for e in evaluation_results if e.passed]
        vetoed = [e for e in evaluation_results if e.vetoed]
        low_score = [e for e in evaluation_results if not e.passed and not e.vetoed]

        summary = {
            "round": round_num,
            "cuisine": cuisine,
            "total_proposals": len(evaluation_results),
            "passed": len(passed),
            "vetoed": len(vetoed),
            "low_score": len(low_score),
            "pass_threshold": pass_threshold,
            "cuisine_sku_count": sku_count,
        }

        # Summarize vetoed items
        if vetoed:
            summary["vetoed_examples"] = [
                {"name": e.proposal.name, "reason": e.veto_reason or "unknown"}
                for e in vetoed[:5]
            ]

        # Summarize low-score items with dimensional breakdown
        if low_score:
            summary["low_score_profile"] = {
                "avg_blue_ocean": f"{sum(e.cuisine_blue_ocean for e in low_score) / len(low_score):.1f}",
                "avg_trend_heat": f"{sum(e.trend_heat for e in low_score) / len(low_score):.1f}",
                "avg_hawker_sub": f"{sum(e.hawker_substitutability for e in low_score) / len(low_score):.1f}",
                "max_score": f"{max(e.total_score for e in low_score):.1f}",
                "examples": [
                    {
                        "name": e.proposal.name,
                        "blue_ocean": e.cuisine_blue_ocean,
                        "trend_heat": e.trend_heat,
                        "hawker_sub": e.hawker_substitutability,
                        "total": e.total_score,
                    }
                    for e in sorted(low_score, key=lambda x: -x.total_score)[:3]
                ],
            }

        system_prompt = """You are a feedback synthesizer in InstaChef's SKU Explorer pipeline.
Your job: compress the evaluator's raw output into a concise, actionable brief for the Generator agent.

Rules:
1. Diagnose WHY proposals failed — identify the dominant failure pattern (veto vs low-score, which dimension is weakest).
2. Translate metrics into CONCRETE guidance the Generator can act on. Don't say "blue ocean score is low" — say "avoid variants of existing dishes; propose niche sub-regional specialties".
3. If multiple proposals were vetoed for the same reason, highlight that pattern once — don't repeat every example.
4. If the pass rate was 0, suggest a direction change rather than incremental tweaks.

Also decide whether to adjust the pass threshold for the next round:
- If failures are ALL vetoes (hard constraints): do NOT lower threshold (vetoed items still fail at any threshold).
- If failures are ALL low scores and the gap is small (avg score within 5 points): lower by 5.
- If failures are ALL low scores and gap is medium (5-10 points): lower by 10.
- If failures are ALL low scores and gap is large (>10 points): lower by 15.
- Floor: never go below 60.
- If some proposals passed already, do NOT lower.

Output JSON:
{"feedback": "<actionable brief for Generator, <300 words>",
 "threshold_adjustment": -5}"""

        user_message = f"""Round {round_num} results for {cuisine} cuisine:

{_json.dumps(summary, ensure_ascii=False, indent=2)}

Output JSON with feedback and threshold_adjustment (negative = lower, 0 = no change)."""

        try:
            from openai import OpenAI
            cfg = self.config["llm"]
            client = OpenAI(
                base_url=cfg.get("base_url", "https://api.siliconflow.cn/v1"),
                api_key=cfg.get("api_key") or _os.getenv(cfg.get("api_key_env", "")),
            )
            response = client.chat.completions.create(
                model=cfg.get("generator_model", cfg.get("evaluator_model", "deepseek-ai/DeepSeek-V3")),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            data = _json.loads(response.choices[0].message.content)
            feedback_text = data.get("feedback", "")
            adjustment = int(data.get("threshold_adjustment", 0))
            return feedback_text.strip(), max(pass_threshold + adjustment, 60)
        except Exception as e:
            # Fallback: basic rule-based feedback + fixed adjustment
            parts = []
            if vetoed:
                reasons = {}
                for v in vetoed:
                    r = v.veto_reason or "unknown"
                    reasons[r] = reasons.get(r, 0) + 1
                parts.append(f"Vetoed: {', '.join(f'{c}x {r}' for r, c in reasons.items())}.")
            if low_score:
                gap = pass_threshold - max(e.total_score for e in low_score)
                if gap <= 5:
                    adj = -5
                elif gap <= 10:
                    adj = -10
                else:
                    adj = -15
                parts.append(
                    f"All {len(low_score)} scored below {pass_threshold}. "
                    f"Try different dish concepts."
                )
                return " ".join(parts), max(pass_threshold + adj, 60)
            return " ".join(parts), pass_threshold

    def _emit(self, cuisine: str, round_num: int, phase: str, meta: dict):
        for cb in self.state_callbacks.get("on_state_change", []):
            cb(cuisine, round_num, phase, meta)

    def _emit_round_complete(self, result: RoundResult):
        for cb in self.state_callbacks.get("on_round_complete", []):
            cb(result)
