import time
import yaml
from models import (
    OrchestratorState, RoundResult, CuisineResult,
    FinalOutput, EvaluationResult, DishProposal, ExecutiveSummary,
    StageTrace, ProcessedCommodity,
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
        t_summary = time.time()
        output.executive_summary = self._generate_executive_summary(output)
        summary_ms = (time.time() - t_summary) * 1000
        print(f"⏱ Pipeline: executive_summary={summary_ms/1000:.1f}s | total={time.time() - t_start:.1f}s", flush=True)

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
        # Trace: search
        t_search = time.time()
        search_results = self.searcher.search_cuisine(cuisine)
        # Save raw search snippets for validation
        self._save_search_snippets(cuisine, search_results)
        trace_search = StageTrace(
            stage="search",
            elapsed_ms=(time.time() - t_search) * 1000,
            model_name="tavily",
            input_size_chars=len(cuisine),
            output_size_chars=sum(len(str(r)) for r in search_results),
        )

        # Summarizing
        self._emit(cuisine, 0, "summarizing", {
            "locked_count": 0,
            "remaining": cfg["target_per_cuisine"],
        })
        # Trace: summarize
        t_summarize = time.time()
        search_summary, ref_map, ref_contents, ref_source_types = self.searcher.summarize_for_generator(
            search_results, cuisine
        )
        trace_summarize = StageTrace(
            stage="summarize",
            elapsed_ms=(time.time() - t_summarize) * 1000,
            model_name=self.config["llm"].get("generator_model", "n/a"),
            input_size_chars=sum(len(str(r)) for r in search_results),
            output_size_chars=len(search_summary),
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

            # Trace: generate
            t_gen = time.time()
            locked_names = [e.proposal.name for e in locked]
            # Inject HITL feedback summary into Generator prompt
            augmented_search = search_summary
            if feedback_summary:
                augmented_search = search_summary + "\n" + feedback_summary

            proposal_dicts = self.generator.generate(
                cuisine=cuisine,
                count=remaining + 5,
                search_summary=augmented_search,
                feedback=feedback,
                locked_names=locked_names,
                round_num=round_num,
            )

            trace_generate = StageTrace(
                stage="generate",
                elapsed_ms=(time.time() - t_gen) * 1000,
                model_name=self.config["llm"].get("generator_model", "n/a"),
                input_size_chars=len(search_summary) + len(feedback),
                output_size_chars=sum(len(str(p)) for p in proposal_dicts) if proposal_dicts else 0,
            )

            if not proposal_dicts:
                break

            # --- HITL Blacklist: deterministic name match on rejected dishes ---
            hitl_vetoed: list[EvaluationResult] = []
            all_rejections: list[dict] = []
            blacklist: set[str] = set()
            feedback_summary = ""
            try:
                from pathlib import Path as _Path
                from utils.feedback_loader import (
                    load_all_rejections, build_blacklist,
                    build_feedback_summary, _normalize_name,
                )
                all_rejections = load_all_rejections(_Path("data/feedback"))
                blacklist = build_blacklist(all_rejections, cuisine)
                feedback_summary = build_feedback_summary(all_rejections, cuisine)
            except Exception:
                pass

            for p in proposal_dicts:
                name = p.get("name", "")
                normalized = _normalize_name(name, sort_tokens=True)
                if normalized and normalized in blacklist:
                    hitl_vetoed.append(EvaluationResult(
                        proposal=DishProposal(
                            id=p.get("id", 0),
                            name=name,
                            name_cn=p.get("name_cn", ""),
                            cuisine=cuisine,
                            price_sgd=p.get("price_sgd", 0),
                            description=p.get("description", ""),
                            description_cn=p.get("description_cn", ""),
                            differentiation=p.get("differentiation", ""),
                            trend_source=p.get("trend_source", ""),
                            source_refs=p.get("source_refs", []),
                        ),
                        vetoed=True,
                        veto_reason="HITL黑名单：确定性去重（此前被人工拒绝）",
                        cuisine_blue_ocean=0,
                        trend_heat=0,
                        hawker_substitutability=0,
                        total_score=0,
                        passed=False,
                        reasoning="HITL 反馈黑名单 — 确定性去重拦截",
                    ))

            # --- Task 4: Source-ref lineage validation (fast-fail hallucinations) ---
            lineage_vetoed: list[EvaluationResult] = []
            validated_proposals: list[dict] = []
            lineage_log: list[dict] = []  # audit trail for lineage_<ts>.json

            for p in proposal_dicts:
                refs = p.get("source_refs", [])
                valid = bool(refs) and all(r in ref_map for r in refs)

                # --- Dual-evidence check ---
                dish_name_matched = False
                trend_matched = False
                evidence_level = "none"
                checked_terms: list[str] = []

                if valid and ref_contents:
                    # Determine best source type among referenced results
                    ref_types = [
                        ref_source_types.get(r, "trend") for r in refs
                        if r in ref_source_types
                    ]
                    has_menu = "menu" in ref_types
                    evidence_level = "menu" if has_menu else "trend"

                    # Dish name match: signature phrase (first 60% words)
                    # Filter out regional/cuisine sub-type names that appear in
                    # every food article but prove nothing about a specific dish
                    _REGION_NOISE = {
                        "sichuan", "hunan", "cantonese", "teochew", "hokkien",
                        "wenzhou", "yangzhou", "lanzhou", "shanghai", "beijing",
                        "fujian", "yunnan", "xinjiang", "guizhou", "jiangxi",
                        "jiangsu", "shaanxi", "hakka", "dongbei", "hainanese",
                        "shanghainese", "chinese", "chongqing", "jiangnan",
                    }
                    # Words so common in food content they prove nothing
                    _COOKING_NOISE = {
                        "with", "dry", "style", "rice", "bowl", "noodles",
                        "sauce", "fresh", "hot", "warm", "served", "dish",
                        "fried", "grilled", "braised", "soup",
                        "beef", "duck", "fish", "tea", "steamed",
                        "chicken", "lamb", "pork", "egg", "tofu",
                    }
                    name = p.get("name", "")
                    words = name.split() or [name]
                    n = max(1, -(-len(words) * 3 // 5))
                    signature_words = [
                        w for w in words[:n]
                        if w.lower().strip(",.()") not in _REGION_NOISE
                    ]
                    # If all words were region noise, fall back to full name
                    if not signature_words:
                        signature_words = words[:n]
                    signature_phrase = " ".join(
                        w.lower().strip(",.()") for w in signature_words
                    )
                    checked_terms.append(f"dish:{signature_phrase}")

                    evidence_hits: list[dict] = []
                    ref_texts = " ".join(
                        ref_contents.get(r, "") for r in refs if r in ref_contents
                    ).lower()

                    if signature_phrase in ref_texts:
                        dish_name_matched = True
                        # Extract snippet around matched phrase
                        idx = ref_texts.find(signature_phrase)
                        start = max(0, idx - 40)
                        end = min(len(ref_texts), idx + len(signature_phrase) + 40)
                        evidence_hits.append({
                            "term": signature_phrase,
                            "snippet": ref_texts[start:end],
                        })
                    else:
                        for w in signature_words:
                            w_clean = w.lower().strip(",.()")
                            if (len(w_clean) >= 3
                                    and w_clean not in _COOKING_NOISE
                                    and w_clean in ref_texts):
                                dish_name_matched = True
                                checked_terms.append(f"partial:{w_clean}")
                                idx = ref_texts.find(w_clean)
                                start = max(0, idx - 40)
                                end = min(len(ref_texts), idx + len(w_clean) + 40)
                                evidence_hits.append({
                                    "term": w_clean,
                                    "snippet": ref_texts[start:end],
                                })
                                break

                    # Trend match: cuisine name, food-type keywords (loose)
                    trend_signals = [
                        cuisine.lower(),
                        cuisine.lower().split("/")[0].strip(),
                        "rice", "bowl", "noodle", "grilled", "braised",
                        "curry", "spicy", "halal",
                    ]
                    for ts in trend_signals:
                        if ts in ref_texts:
                            trend_matched = True
                            checked_terms.append(f"trend:{ts}")
                            break  # one trend signal is enough

                # Determine if any referenced source came from Hop 2 (menu deep-search)
                hop2_count = sum(
                    1 for r in refs
                    if r in ref_contents and ref_source_types.get(r) == "menu"
                )

                lineage_log.append({
                    "name": p.get("name", ""),
                    "name_cn": p.get("name_cn", ""),
                    "source_refs": refs,
                    "validated": valid,
                    "evidence_level": evidence_level,
                    "dish_name_matched": dish_name_matched,
                    "trend_matched": trend_matched,
                    "hop2_refs": hop2_count,
                    "checked_terms": checked_terms,
                    "evidence_hits": evidence_hits,
                    "ref_urls": {r: ref_map[r] for r in refs if r in ref_map} if valid else {},
                })
                if not valid:
                    lineage_vetoed.append(EvaluationResult(
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
                            source_refs=p.get("source_refs", []),
                        ),
                        vetoed=True,
                        veto_reason="幻觉数据：未引用有效搜索来源（source_refs 为空或包含无效引用标签）",
                        cuisine_blue_ocean=0,
                        trend_heat=0,
                        hawker_substitutability=0,
                        total_score=0,
                        passed=False,
                        reasoning="数据谱系验证失败 — Fast-Failure 拦截",
                    ))
                else:
                    # Inject ref_map URLs for traceability
                    p["_ref_urls"] = {r: ref_map[r] for r in refs}
                    validated_proposals.append(p)

            # --- Task 1: Hard-coded guard pre-check (before Evaluator LLM) ---
            guard_vetoed: list[EvaluationResult] = []
            guard_passed_proposals: list[dict] = []

            for p in validated_proposals:
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
                            source_refs=p.get("source_refs", []),
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

            # Trace: evaluate
            t_eval = time.time()
            # Build HITL RAG context for this cuisine
            cuisine_rejections = [
                r for r in all_rejections if r.get("cuisine") == cuisine
            ]
            hitl_context = ""
            if cuisine_rejections:
                lines = [
                    "\n## HITL Feedback: Previously Rejected Proposals",
                    "These dishes were rejected by human experts. Be extra vigilant for similar patterns:\n"
                ]
                for r in cuisine_rejections[-10:]:
                    lines.append(
                        f"- 「{r.get('proposal_name_cn', r.get('proposal_name', ''))}」"
                        f"→ {r.get('reason_label', '')}"
                    )
                hitl_context = "\n".join(lines)

            if guard_passed_proposals:
                evaluations_raw, summary, suggestions = self.evaluator.evaluate(
                    proposals=guard_passed_proposals,
                    existing_skus=cuisine_skus,
                    round_num=round_num,
                    locked_count=len(locked),
                    remaining=remaining,
                    pass_threshold=current_threshold,
                    hitl_context=hitl_context,
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

            trace_evaluate = StageTrace(
                stage="evaluate",
                elapsed_ms=(time.time() - t_eval) * 1000,
                model_name=self.config["llm"].get("evaluator_model", "n/a"),
                input_size_chars=sum(len(str(p)) for p in guard_passed_proposals),
                output_size_chars=sum(len(str(e)) for e in llm_results),
            )

            # Combine: HITL-vetoed + lineage-vetoed + guard-vetoed + LLM-evaluated
            evaluation_results = hitl_vetoed + lineage_vetoed + guard_vetoed + llm_results

            # JUDGING — apply evidence penalty before pass/fail
            self.state = OrchestratorState.JUDGING
            for i, e in enumerate(evaluation_results):
                if not e.vetoed and i < len(lineage_log):
                    li = lineage_log[i]
                    if not li.get("dish_name_matched"):
                        # Weak evidence: dish name not found in search content
                        # Check if trend_match is only generic (cuisine name)
                        terms = li.get("checked_terms", [])
                        generic_only = all(
                            t.startswith("trend:") for t in terms
                        ) if terms else True
                        penalty = 10 if generic_only else 5
                        e.total_score = max(0, e.total_score - penalty)
                        e.passed = e.total_score >= current_threshold
                        e.reasoning += (
                            f" [证据降权-{penalty}: dish_name未匹配"
                            + ("且仅有泛化趋势信号" if generic_only else "")
                            + "]"
                        )

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
                ref_map=ref_map,
                lineage_results=lineage_log,
            )
            rounds_history.append(round_result)
            self._emit_round_complete(round_result)

            # --- Trace: feedback ---
            t_fb = time.time()

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

            trace_feedback = StageTrace(
                stage="feedback",
                elapsed_ms=(time.time() - t_fb) * 1000,
                model_name=self.config["llm"].get("generator_model", "n/a"),
                input_size_chars=sum(len(str(e)) for e in evaluation_results),
                output_size_chars=len(feedback),
            )

            # Collect all stage traces for this round
            round_traces = [trace_generate, trace_evaluate, trace_feedback]
            if round_num == 1:
                round_traces = [trace_search, trace_summarize] + round_traces

            round_result.stage_traces = round_traces

            # Print compact trace summary
            trace_parts = []
            for t in round_traces:
                sec = t.elapsed_ms / 1000
                trace_parts.append(f"{t.stage}={sec:.1f}s")
            print(f"⏱ {cuisine} Round {round_num} traces: {' | '.join(trace_parts)}", flush=True)

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

    def _generate_executive_summary(self, output: FinalOutput) -> ExecutiveSummary:
        """LLM-generated business-readable summary for stakeholders."""
        import json as _json
        import os as _os

        # Build compact aggregate stats
        total_proposals = 0
        total_passed = 0
        veto_patterns: dict[str, int] = {}
        low_score_notes: list[str] = []

        for cuisine, cr in output.cuisines.items():
            for rr in cr.rounds_history:
                total_proposals += rr.proposals_generated
                total_passed += sum(1 for e in rr.evaluations if e.passed)
                for e in rr.evaluations:
                    if e.vetoed and e.veto_reason:
                        key = e.veto_reason[:60]
                        veto_patterns[key] = veto_patterns.get(key, 0) + 1
                    elif not e.passed and not e.vetoed:
                        low_score_notes.append(
                            f"{e.proposal.name}: blue={e.cuisine_blue_ocean:.0f} "
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
- Overall pass rate: {pass_rate}
- Total proposals evaluated: {total_proposals}
- Top veto patterns: {_json.dumps(top_vetoes)}
- Low-score notes (sample): {_json.dumps(low_score_notes[:5])}
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
            from openai import OpenAI
            cfg = self.config["llm"]
            client = OpenAI(
                base_url=cfg.get("base_url", "https://api.siliconflow.cn/v1"),
                api_key=cfg.get("api_key") or _os.getenv(cfg.get("api_key_env", "")),
            )
            response = client.chat.completions.create(
                model=cfg.get("generator_model", "deepseek-ai/DeepSeek-V3"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"},
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

    def _save_search_snippets(self, cuisine: str, results: list[dict]):
        """Persist raw Tavily search snippets for validation."""
        import json as _json
        from datetime import datetime
        from pathlib import Path

        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"search_snippets_{ts}.json"

        snippets = []
        for r in results:
            snippets.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
                "source_type": r.get("source_type", "trend"),
                "evidence_level": r.get("evidence_level", "trend"),
            })

        with open(path, "w") as f:
            _json.dump({
                "timestamp": ts,
                "cuisine": cuisine,
                "total_results": len(snippets),
                "snippets": snippets,
            }, f, ensure_ascii=False, indent=2)

    def _emit(self, cuisine: str, round_num: int, phase: str, meta: dict):
        for cb in self.state_callbacks.get("on_state_change", []):
            cb(cuisine, round_num, phase, meta)

    def _emit_round_complete(self, result: RoundResult):
        for cb in self.state_callbacks.get("on_round_complete", []):
            cb(result)
