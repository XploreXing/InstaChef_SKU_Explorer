import json
import os
from openai import OpenAI
from models import DishProposal, EvaluationResult


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
    return [
        {
            "id": c.id, "name": c.name, "description": c.description,
            "cuisine_type": c.cuisine_type,
            "is_halal_suspect": c.is_halal_suspect,
            "is_fried": c.is_fried,
        }
        for c in commodities
    ]


def build_evaluator_user_message(
    proposals: list[dict],
    existing_skus: list,
    round_num: int,
    locked_count: int,
    remaining: int,
) -> str:
    parts = [
        f"## Current InstaChef SKU Catalog ({len(existing_skus)} total SKUs)",
        json.dumps(_serialize_commodities(existing_skus), indent=2, ensure_ascii=False),
        "",
        "## Proposals to Evaluate",
        json.dumps(proposals, indent=2, ensure_ascii=False),
        "",
        "## Progress",
        f"Round {round_num} of 3 for this cuisine.",
        f"Currently {locked_count} accepted, need {remaining} more.",
        "",
        "Evaluate each proposal against the 4 hard constraints first.",
        "If any constraint fails, mark as vetoed with score 0.",
        "Only score the 3 dimensions if ALL constraints pass.",
    ]
    return "\n".join(parts)


class EvaluatorAgent:
    def __init__(self, config: dict):
        self.cfg = config["llm"]
        self.client = OpenAI(
            base_url=self.cfg["base_url"],
            api_key=os.getenv(self.cfg["api_key_env"]),
        )
        self.system_prompt = _load_system_prompt()

    def evaluate(
        self,
        proposals: list[dict],
        existing_skus: list,
        round_num: int = 1,
        locked_count: int = 0,
        remaining: int = 10,
    ) -> tuple[list[dict], dict, str]:
        user_message = build_evaluator_user_message(
            proposals=proposals,
            existing_skus=existing_skus,
            round_num=round_num,
            locked_count=locked_count,
            remaining=remaining,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.cfg["evaluator_model"],
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.cfg["evaluator_temperature"],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return self._parse_response(content)
        except Exception as e:
            print(f"Evaluator API call failed: {e}")
            return [], {}, ""

    def _parse_response(self, content: str) -> tuple[list[dict], dict, str]:
        try:
            data = json.loads(content)
            evaluations = data.get("evaluations", [])
            summary = data.get("summary", {})
            suggestions = data.get("improvement_suggestions", "")
            return evaluations, summary, suggestions
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse Evaluator response: {e}")
            return [], {}, ""

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
                cuisine=ev.get("cuisine", prop_dict.get("cuisine", "")),
                price_sgd=prop_dict.get("price_sgd", 0.0),
                description=prop_dict.get("description", ""),
                differentiation=prop_dict.get("differentiation", ""),
                trend_source=prop_dict.get("trend_source", ""),
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
