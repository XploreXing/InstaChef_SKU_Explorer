import time
import yaml
from models import (
    OrchestratorState, RoundResult, CuisineResult,
    FinalOutput, EvaluationResult, ProcessedCommodity,
)
from utils.data_loader import SKUDataLoader
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

        cuisine_skus = self.sku_loader.get_by_cuisine(cuisine)

        search_results = self.searcher.search_cuisine(cuisine)
        search_summary = self.searcher.summarize_for_generator(
            search_results, cuisine
        )

        for round_num in range(1, cfg["max_rounds_per_cuisine"] + 1):
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

            # EVALUATING
            self.state = OrchestratorState.EVALUATING
            self._emit(cuisine, round_num, "evaluating", {
                "proposal_count": len(proposal_dicts)
            })

            evaluations_raw, summary, suggestions = self.evaluator.evaluate(
                proposals=proposal_dicts,
                existing_skus=self.existing_skus,
                round_num=round_num,
                locked_count=len(locked),
                remaining=remaining,
            )

            if not evaluations_raw:
                break

            evaluation_results = EvaluatorAgent.to_evaluation_results(
                evaluations_raw, proposal_dicts
            )

            # JUDGING
            self.state = OrchestratorState.JUDGING
            new_passed = [e for e in evaluation_results if e.passed]
            new_rejected = [e for e in evaluation_results if not e.passed]
            locked.extend(new_passed)

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

            feedback = self._extract_improvement_feedback(
                summary.get("rejection_reasons", []), suggestions
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

    def _extract_improvement_feedback(
        self, rejection_reasons: list[dict], suggestions: str
    ) -> str:
        parts = []
        if rejection_reasons:
            parts.append("Rejected proposals and reasons:")
            for r in rejection_reasons[:5]:
                parts.append(f"- {r.get('name', '?')}: {r.get('reason', 'no reason')}")
        if suggestions:
            parts.append(f"\nSuggestions: {suggestions}")
        return "\n".join(parts) if parts else ""

    def _emit(self, cuisine: str, round_num: int, phase: str, meta: dict):
        for cb in self.state_callbacks.get("on_state_change", []):
            cb(cuisine, round_num, phase, meta)

    def _emit_round_complete(self, result: RoundResult):
        for cb in self.state_callbacks.get("on_round_complete", []):
            cb(result)
