import time
import yaml
from models import (
    OrchestratorState, RoundResult, CuisineResult,
    FinalOutput, EvaluationResult, DishProposal, ExecutiveSummary,
    StageTrace, ProcessedCommodity,
)
from utils.data_loader import SKUDataLoader
from utils.guard import HardConstraintGuard
from utils.feedback_loader import _normalize_name
from utils.observability import ObservabilityLogger
from utils.quota import allocate_quota
from agents.generator import GeneratorAgent
from agents.evaluator import EvaluatorAgent
import threading
from utils.duplication_checker import check_duplicates


def _vetoed_result(p: dict, cuisine: str, veto_reason: str, reasoning: str) -> EvaluationResult:
    """A proposal stopped by a deterministic layer, before the evaluator sees it."""
    return EvaluationResult(
        proposal=DishProposal(
            id=p.get("id", 0),
            name=p.get("name", ""),
            name_cn=p.get("name_cn", ""),
            cuisine=cuisine,
            price_sgd=p.get("price_sgd", 0),
            description=p.get("description", ""),
            description_cn=p.get("description_cn", ""),
            differentiation=p.get("differentiation", ""),
            trend_source=p.get("trend_source", ""),
            source_urls=p.get("source_urls", []),
        ),
        vetoed=True,
        veto_reason=veto_reason,
        trend_heat=0,
        hawker_substitutability=0,
        total_score=0,
        passed=False,
        reasoning=reasoning,
    )


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
        self.evaluator: EvaluatorAgent | None = None
        self._eval_lock: threading.Lock | None = None
        self.breaker = None  # CircuitBreaker, injected by app.py (or self-created)
        self._summary_router = None  # cached LLMRouter for feedback/summary calls

        self.obs: "ObservabilityLogger | None" = None  # set by app.py or tests

    def load_skus(self):
        self.sku_loader = SKUDataLoader(self.config, breaker=self.breaker)
        self.existing_skus = self.sku_loader.load()
        self.state = OrchestratorState.INIT #如果改为外部并发，即多个进程同时启动_process_cuisine是，因为这个state当前状态对并发完全没意义，只有内部的locked,round_num这些内部变量可以充当状态

    def _init_agents(self):
        if self.evaluator is None:
            self.evaluator = EvaluatorAgent(self.config, breaker=self.breaker)
        self._eval_lock = threading.Lock()

    def _get_summary_router(self):
        """Lazy-init the LLMRouter used for feedback synthesis + executive
        summary. Shares the same breaker as the agents for coherent cooldown."""
        if self._summary_router is None:
            from utils.llm_router import LLMRouter
            self._summary_router = LLMRouter(
                self.config["llm"],
                role="generator",  # reuse generator model for summary tasks
                selected_preset_id=self.config["llm"].get("selected_preset_id"),
                breaker=self.breaker,
            )
        return self._summary_router


    def run(self, cuisines: list[str] | None = None,
            targets: dict[str, int] | None = None) -> FinalOutput:
        """Look for dishes in `cuisines`. `targets` says how many per cuisine;
        without it, orchestrator.total_target is split across the cuisines by
        how few SKUs each already has. A cuisine with a quota of 0 is skipped."""
        cfg = self.config["orchestrator"]
        selected = cuisines or cfg["cuisines"]
        if targets is None:
            counts = self.sku_loader.get_cuisine_counts()
            targets = allocate_quota(
                cfg["total_target"], {c: counts.get(c, 0) for c in selected}
            )
        targets = {c: targets[c] for c in selected if targets.get(c, 0) > 0}

        self._init_agents()
        t_start = time.time()
        self.results={}
        if targets:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(targets)) as pool:
                futures={
                    pool.submit(self._process_cuisine, c, n) :c for c, n in targets.items()
                }
                for future in as_completed(futures):
                    cuisine=futures[future]
                    try:
                        self.results[cuisine]=future.result()
                    except Exception as e:
                        print(f'{cuisine}failed : {e}',flush=True)
                        self.results[cuisine]=CuisineResult(
                            cuisine=cuisine,total_rounds=0,locked=[],rounds_history=[],
                        )

        self.state = OrchestratorState.AGGREGATING
        output = FinalOutput(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            cuisines=self.results,
            total_elapsed_seconds=time.time() - t_start,
        )
        t_summary = time.time()
        output.executive_summary = self._generate_executive_summary(output)
        summary_ms = (time.time() - t_summary) * 1000
        print(f"⏱ Pipeline: executive_summary={summary_ms/1000:.1f}s | total={time.time() - t_start:.1f}s", flush=True)

        self.state = OrchestratorState.DONE
        return output

    def _process_cuisine(self, cuisine: str, target: int) -> CuisineResult:
        cfg = self.config["orchestrator"]
        #Each thread gets its own Generator (Thread-safe ToolContext)
        generator=GeneratorAgent(self.config, breaker=self.breaker)
        evaluator=self.evaluator

        # Every evaluated dish that was not vetoed. The best `target` of them
        # are selected at the end; there is no score a dish has to reach.
        candidates: list[EvaluationResult] = []
        rounds_history: list[RoundResult] = []
        feedback = ""

        cuisine_skus = self.sku_loader.get_by_cuisine(cuisine)
        max_rounds = cfg["max_rounds_per_cuisine"]

        for round_num in range(1, max_rounds + 1):
            if self.stop_check and self.stop_check():
                break
            remaining = target - len(candidates)
            if remaining <= 0:
                break

            t_round = time.time()

            # GENERATING
            self.state = OrchestratorState.GENERATING
            self._emit(cuisine, round_num, "generating", {
                "locked_count": len(candidates), "remaining": remaining
            })

            # Load HITL feedback (must be before Generator call)
            blacklist: set[str] = set()
            feedback_summary = ""
            all_rejections: list[dict] = []
            try:
                from pathlib import Path as _Path
                from utils.feedback_loader import (
                    load_all_rejections, build_blacklist,
                    build_feedback_summary, build_evaluator_context,
                )
                all_rejections = load_all_rejections(_Path("data/feedback"))
                blacklist = build_blacklist(all_rejections, cuisine)
                feedback_summary = build_feedback_summary(all_rejections, cuisine)
            except Exception:
                pass

            # Trace: generate
            t_gen = time.time()
            locked_names = [e.proposal.name for e in candidates]

            print(f"  [generate] cuisine={cuisine}, round={round_num}, existing_skus={len(cuisine_skus)}", flush=True)
            proposal_dicts = generator.generate(
                cuisine=cuisine,
                count=remaining + 5,
                feedback=feedback,
                HITL_feedback=feedback_summary,
                locked_names=locked_names,
                round_num=round_num,
                existing_skus=cuisine_skus,
            )

            trace_generate = StageTrace(
                stage="generate",
                elapsed_ms=(time.time() - t_gen) * 1000,
                model_name=self.config["llm"].get("generator_model", "n/a"),
                input_size_chars=len(feedback)+len(feedback_summary),
                output_size_chars=sum(len(str(p)) for p in proposal_dicts) if proposal_dicts else 0,
            )
            self._obs_emit(
                cuisine=cuisine, round_num=round_num, stage="generate",
                event_type="stage_end", status="success",
                elapsed_ms=trace_generate.elapsed_ms,
                model_name=trace_generate.model_name,
                input_size_chars=trace_generate.input_size_chars,
                output_size_chars=trace_generate.output_size_chars,
                payload={
                    "proposals_count": len(proposal_dicts) if proposal_dicts else 0,
                    "hitl_feedback_injected": bool(feedback_summary),
                    "dynamic_feedback_injected": bool(feedback),
                },
            )

            if not proposal_dicts:
                break

            # --- HITL Blacklist: deterministic name match (uses pre-loaded blacklist) ---
            hitl_vetoed: list[EvaluationResult] = []
            hitl_passed_proposals: list[dict] = []

            for p in proposal_dicts:
                name = p.get("name", "")
                normalized = _normalize_name(name, sort_tokens=True)
                if normalized and normalized in blacklist:
                    hitl_vetoed.append(_vetoed_result(
                        p, cuisine,
                        veto_reason="HITL黑名单：确定性去重（此前被人工拒绝）",
                        reasoning="HITL 反馈黑名单 — 确定性去重拦截",
                    ))
                    self._obs_emit(
                        cuisine=cuisine, round_num=round_num, stage="generate",
                        event_type="veto", status="success",
                        payload={
                            "proposal_name": name,
                            "proposal_name_cn": p.get("name_cn", ""),
                            "veto_layer": "hitl_blacklist",
                            "veto_reason": "HITL黑名单：确定性去重（此前被人工拒绝）",
                            "immediate": True,
                        },
                    )
                else:
                    hitl_passed_proposals.append(p)
            #不再需要在Orchestrator的逻辑中进行lineage validation逻辑


            # --- Task 1: Hard-coded guard pre-check (before Evaluator LLM) ---
            guard_vetoed: list[EvaluationResult] = []
            guard_passed_proposals: list[dict] = []

            for p in hitl_passed_proposals:
                passed, veto_reason = HardConstraintGuard.precheck(p)
                if not passed:
                    guard_vetoed.append(_vetoed_result(
                        p, p.get("cuisine", ""),
                        veto_reason=veto_reason,
                        reasoning="高压线熔断 — 无需 LLM 评估",
                    ))
                    self._obs_emit(
                        cuisine=cuisine, round_num=round_num, stage="evaluate",
                        event_type="veto", status="success",
                        payload={
                            "proposal_name": p.get("name", ""),
                            "proposal_name_cn": p.get("name_cn", ""),
                            "veto_layer": "guard",
                            "veto_reason": veto_reason,
                            "immediate": True,
                        },
                    )
                else:
                    guard_passed_proposals.append(p)

            dup_vetoed: list[EvaluationResult] = []
            dup_passed_proposals: list[dict] = []
            for p in guard_passed_proposals:
                is_dup, reason, score = check_duplicates(
                    name=p.get("name", ""),
                    name_cn=p.get("name_cn", ""),
                    existing_skus=cuisine_skus
    )
                if is_dup:
                    dup_vetoed.append(_vetoed_result(
                        p, p.get("cuisine", ""),
                        veto_reason=reason,
                        reasoning=f"去重检查 — 确定性拦截 (相似度 {score:.0%})",
                    ))
                    self._obs_emit(
                        cuisine=cuisine, round_num=round_num, stage="evaluate",
                        event_type="veto", status="success",
                        payload={
                            "proposal_name": p.get("name", ""),
                            "proposal_name_cn": p.get("name_cn", ""),
                            "veto_layer": "duplicate",
                            "veto_reason": reason,
                            "similarity": round(score, 3),
                            "immediate": True,
                        },
                    )
                else:
                    dup_passed_proposals.append(p)

            # EVALUATING — only guard-passed proposals go to LLM
            self.state = OrchestratorState.EVALUATING
            self._emit(cuisine, round_num, "evaluating", {
                "proposal_count": len(proposal_dicts),
                "guard_vetoed": len(guard_vetoed),
                "locked_count": len(candidates),
                "remaining": remaining,
            })

            # Trace: evaluate
            t_eval = time.time()
            # Build HITL RAG context for this cuisine
            hitl_context = ""
            if all_rejections:
                hitl_context = build_evaluator_context(all_rejections, cuisine)

            if dup_passed_proposals:
                with self._eval_lock:
                    evaluations_raw, summary, suggestions = evaluator.evaluate(
                        proposals=dup_passed_proposals,
                        hitl_context=hitl_context,
                )
                # A dish the evaluator did not score comes back vetoed with that
                # as the reason, so an evaluator that returns nothing is visible
                # in the round instead of looking like an empty batch.
                llm_results = EvaluatorAgent.to_evaluation_results(
                    evaluations_raw, dup_passed_proposals
                )
            else:
                llm_results = []
                suggestions = ""

            trace_evaluate = StageTrace(
                stage="evaluate",
                elapsed_ms=(time.time() - t_eval) * 1000,
                model_name=self.config["llm"].get("evaluator_model", "n/a"),
                input_size_chars=sum(len(str(p)) for p in dup_passed_proposals),
                output_size_chars=sum(len(str(e)) for e in llm_results),
            )
            eval_vetoed = [e for e in llm_results if e.vetoed]
            new_candidates = [e for e in llm_results if e.passed]
            self._obs_emit(
                cuisine=cuisine, round_num=round_num, stage="evaluate",
                event_type="stage_end", status="success",
                elapsed_ms=trace_evaluate.elapsed_ms,
                model_name=trace_evaluate.model_name,
                input_size_chars=trace_evaluate.input_size_chars,
                output_size_chars=trace_evaluate.output_size_chars,
                payload={
                    "evaluated_count": len(dup_passed_proposals),
                    "candidate_count": len(new_candidates),
                    "vetoed_count": len(eval_vetoed),
                    "avg_score": round(sum(e.total_score for e in new_candidates) / max(len(new_candidates), 1), 1),
                },
            )

            # Combine: HITL-vetoed + guard-vetoed + dup-vetoed + LLM-evaluated
            evaluation_results = hitl_vetoed + guard_vetoed + dup_vetoed + llm_results
            candidates.extend(new_candidates)

            # Observability: judging stage summary
            self._obs_emit(
                cuisine=cuisine, round_num=round_num, stage="judging",
                event_type="stage_end", status="success",
                payload={
                    "target": target,
                    "new_candidates": len(new_candidates),
                    "total_candidates": len(candidates),
                },
            )
            self._emit(cuisine, round_num, "judging", {
                "locked_count": min(len(candidates), target),
                "remaining": max(0, target - len(candidates)),
            })

            round_result = RoundResult(
                cuisine=cuisine,
                round_num=round_num,
                proposals_generated=len(proposal_dicts),
                passed_count=len(new_candidates),
                rejected_count=len(evaluation_results) - len(new_candidates),
                locked_total=min(len(candidates), target),
                evaluations=evaluation_results,
                improvement_suggestions=suggestions,
                elapsed_seconds=time.time() - t_round,
                ref_map={},
                lineage_results=[],
            )
            rounds_history.append(round_result)
            self._emit_round_complete(round_result)

            round_traces = [trace_generate, trace_evaluate]

            # Feedback only matters if another round is going to use it.
            if len(candidates) < target and round_num < max_rounds:
                t_fb = time.time()
                # --- LLM semantic compression: brief the Generator for the next round ---
                feedback = self._synthesize_feedback(
                    evaluation_results=evaluation_results,
                    cuisine=cuisine,
                    round_num=round_num,
                )
                trace_feedback = StageTrace(
                    stage="feedback",
                    elapsed_ms=(time.time() - t_fb) * 1000,
                    model_name=self.config["llm"].get("generator_model", "n/a"),
                    input_size_chars=sum(len(str(e)) for e in evaluation_results),
                    output_size_chars=len(feedback),
                )
                self._obs_emit(
                    cuisine=cuisine, round_num=round_num, stage="feedback",
                    event_type="stage_end", status="success",
                    elapsed_ms=trace_feedback.elapsed_ms,
                    model_name=trace_feedback.model_name,
                    input_size_chars=trace_feedback.input_size_chars,
                    output_size_chars=len(feedback),
                    payload={"feedback_length_chars": len(feedback)},
                )
                round_traces.append(trace_feedback)

            round_result.stage_traces = round_traces

            # Print compact trace summary
            trace_parts = []
            for t in round_traces:
                sec = t.elapsed_ms / 1000
                trace_parts.append(f"{t.stage}={sec:.1f}s")
            print(f"⏱ {cuisine} Round {round_num} traces: {' | '.join(trace_parts)}", flush=True)

        # Selection: the best `target` candidates by score. sorted() is stable,
        # so equal scores keep the order they were proposed in.
        locked = sorted(candidates, key=lambda e: e.total_score, reverse=True)[:target]
        return CuisineResult(
            cuisine=cuisine,
            total_rounds=len(rounds_history),
            locked=locked,
            rounds_history=rounds_history,
        )

    def _synthesize_feedback(
        self,
        evaluation_results: list[EvaluationResult],
        cuisine: str,
        round_num: int,
    ) -> str:
        """LLM-based semantic compression: summarize evaluator findings and
        produce actionable feedback for the Generator's next round."""
        import json as _json

        # Build a compact round summary for the LLM
        vetoed = [e for e in evaluation_results if e.vetoed]
        ranked = sorted(
            (e for e in evaluation_results if e.passed),
            key=lambda e: e.total_score, reverse=True,
        )

        def _scores(e: EvaluationResult) -> dict:
            return {
                "name": e.proposal.name,
                "trend_heat": e.trend_heat,
                "hawker_sub": e.hawker_substitutability,
                "total": e.total_score,
            }

        summary = {
            "round": round_num,
            "cuisine": cuisine,
            "total_proposals": len(evaluation_results),
            "usable": len(ranked),
            "vetoed": len(vetoed),
        }

        # Summarize vetoed items
        if vetoed:
            summary["vetoed_examples"] = [
                {"name": e.proposal.name, "reason": e.veto_reason or "unknown"}
                for e in vetoed[:5]
            ]

        # Dimension scores of the best and the worst usable dishes
        if ranked:
            summary["strongest"] = [_scores(e) for e in ranked[:3]]
            summary["weakest"] = [_scores(e) for e in ranked[3:][-3:]]

        system_prompt = """You are a feedback synthesizer in InstaChef's SKU Explorer pipeline.
Your job: compress the evaluator's raw output into a concise, actionable brief for the Generator agent, which will propose more dishes for this cuisine.

Rules:
1. Diagnose what went wrong — the dominant veto reason, and which dimension is weakest among the lowest-scoring dishes.
2. Translate metrics into CONCRETE guidance the Generator can act on. Don't say "trend heat is low" — say "propose dishes that restaurant chains in Singapore are already promoting".
3. If multiple proposals were vetoed for the same reason, highlight that pattern once — don't repeat every example.
4. If nothing was usable, suggest a direction change rather than incremental tweaks.

Output JSON:
{"feedback": "<actionable brief for Generator, <300 words>"}"""

        user_message = f"""Round {round_num} results for {cuisine} cuisine:

{_json.dumps(summary, ensure_ascii=False, indent=2)}

Output JSON with feedback."""

        try:
            client = self._get_summary_router()
            response = client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=600,
                response_format={"type": "json_object"},
                thinking=False,  # reasoning tokens would eat the 600-token budget
            )
            data = _json.loads(response.choices[0].message.content)
            return data.get("feedback", "").strip()
        except Exception as e:
            # Fallback: basic rule-based feedback
            parts = []
            if vetoed:
                reasons = {}
                for v in vetoed:
                    r = v.veto_reason or "unknown"
                    reasons[r] = reasons.get(r, 0) + 1
                parts.append(f"Vetoed: {', '.join(f'{c}x {r}' for r, c in reasons.items())}.")
            if not ranked:
                parts.append("No proposal was usable. Try different dish concepts.")
            return " ".join(parts)

    def _generate_executive_summary(self, output: FinalOutput) -> ExecutiveSummary:
        """LLM-generated business-readable summary for stakeholders."""
        import json as _json

        # Build compact aggregate stats
        total_proposals = 0
        total_passed = 0
        veto_patterns: dict[str, int] = {}
        low_score_notes: list[str] = []

        for cuisine, cr in output.cuisines.items():
            total_passed += len(cr.locked)
            selected = {id(e) for e in cr.locked}
            for rr in cr.rounds_history:
                total_proposals += rr.proposals_generated
                for e in rr.evaluations:
                    if e.vetoed and e.veto_reason:
                        key = e.veto_reason[:60]
                        veto_patterns[key] = veto_patterns.get(key, 0) + 1
                    elif not e.vetoed and id(e) not in selected:
                        low_score_notes.append(
                            f"{e.proposal.name}: "
                            f"trend={e.trend_heat:.0f} hawker={e.hawker_substitutability:.0f}"
                        )

        pass_rate = f"{total_passed}/{total_proposals} ({100*total_passed/max(total_proposals,1):.0f}%)"
        top_vetoes = [f"{c}x {r}" for r, c in sorted(veto_patterns.items(), key=lambda x: -x[1])[:5]]
        top_passed = [
            e.proposal.name
            for cr in output.cuisines.values()
            for e in cr.locked[:3]
        ]

        system_prompt = """You are an executive summarizer for InstaChef's SKU Explorer.
Generate a concise business summary in JSON format. Keep it under 200 words total.
Focus on patterns and actionable insights, not raw data."""

        user_message = f"""Pipeline completed. Aggregate stats:
- Selected / proposed: {pass_rate}
- Total proposals evaluated: {total_proposals}
- Top veto patterns: {_json.dumps(top_vetoes)}
- Usable but not selected (sample): {_json.dumps(low_score_notes[:5])}
- Top locked recommendations: {_json.dumps(top_passed)}

Output JSON:
{{"overall_pass_rate": "{pass_rate}",
  "total_proposals": {total_proposals},
  "total_passed": {total_passed},
  "veto_patterns": [...],
  "low_score_patterns": [...],
  "risk_direction": "...",
  "top_recommendations": [...]}}"""

        try:
            cfg = self.config["llm"]
            client = self._get_summary_router()
            response = client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"},
                thinking=False,  # reasoning tokens would eat the 500-token budget
            )
            data = _json.loads(response.choices[0].message.content)
            return ExecutiveSummary(
                overall_pass_rate=data.get("overall_pass_rate", pass_rate),
                total_proposals=data.get("total_proposals", total_proposals),
                total_passed=data.get("total_passed", total_passed),
                veto_patterns=data.get("veto_patterns", top_vetoes),
                low_score_patterns=data.get("low_score_patterns", []),
                risk_direction=data.get("risk_direction", ""),
                top_recommendations=data.get("top_recommendations", top_passed),
            )
        except Exception:
            return ExecutiveSummary(
                overall_pass_rate=pass_rate,
                total_proposals=total_proposals,
                total_passed=total_passed,
                veto_patterns=top_vetoes,
                low_score_patterns=[],
                risk_direction="(LLM summary unavailable)",
                top_recommendations=top_passed,
            )
    #不再需要保存搜索的snippet逻辑


    def _obs_emit(self, **kwargs):
        """Convenience: emit to self.obs if set, no-op otherwise."""
        if self.obs is not None:
            self.obs.emit(**kwargs)

    def _emit(self, cuisine: str, round_num: int, phase: str, meta: dict):
        for cb in self.state_callbacks.get("on_state_change", []): # on_state_chage是hook事件
            cb(cuisine, round_num, phase, meta)

    def _emit_round_complete(self, result: RoundResult):
        for cb in self.state_callbacks.get("on_round_complete", []): #on_round_complete也是hook事件
            cb(result)
