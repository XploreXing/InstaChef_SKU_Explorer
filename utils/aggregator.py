import json
from pathlib import Path
from models import EvaluationResult, FinalOutput


class Aggregator:
    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def deduplicate(evaluations: list[EvaluationResult]) -> list[EvaluationResult]:
        seen_names = set()
        unique = []
        for e in evaluations:
            name_lower = e.proposal.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique.append(e)
        return unique

    @staticmethod
    def sort_by_score(evaluations: list[EvaluationResult]) -> list[EvaluationResult]:
        return sorted(evaluations, key=lambda e: e.total_score, reverse=True)

    @staticmethod
    def filter_passed(evaluations: list[EvaluationResult]) -> list[EvaluationResult]:
        return [e for e in evaluations if e.passed]

    def write_output(self, output: FinalOutput) -> str:
        cfg = self.config["output"]
        base = Path(cfg["base_dir"])
        base.mkdir(parents=True, exist_ok=True)
        filename = f"{cfg['filename_prefix']}_{output.timestamp[:10]}.json"
        filepath = base / filename
        with open(filepath, "w") as f:
            json.dump(self._serialize(output), f, indent=2, ensure_ascii=False, default=str)
        return str(filepath)

    def _serialize(self, output: FinalOutput) -> dict:
        return {
            "timestamp": output.timestamp,
            "cuisines": {
                cuisine: {
                    "cuisine": cr.cuisine,
                    "total_rounds": cr.total_rounds,
                    "locked_count": len(cr.locked),
                    "locked": [
                        {
                            "name": e.proposal.name,
                            "cuisine": e.proposal.cuisine,
                            "price_sgd": e.proposal.price_sgd,
                            "description": e.proposal.description,
                            "differentiation": e.proposal.differentiation,
                            "trend_source": e.proposal.trend_source,
                            "total_score": e.total_score,
                            "cuisine_blue_ocean": e.cuisine_blue_ocean,
                            "trend_heat": e.trend_heat,
                            "hawker_substitutability": e.hawker_substitutability,
                        }
                        for e in cr.locked
                    ],
                }
                for cuisine, cr in output.cuisines.items()
            },
            "total_elapsed_seconds": output.total_elapsed_seconds,
        }
