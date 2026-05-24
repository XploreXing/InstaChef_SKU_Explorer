from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OrchestratorState(Enum):
    IDLE = "idle"
    INIT = "init"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    JUDGING = "judging"
    AGGREGATING = "aggregating"
    DONE = "done"
    ERROR = "error"


@dataclass
class DishProposal:
    id: int
    name: str
    cuisine: str
    price_sgd: float
    description: str
    differentiation: str
    trend_source: str


@dataclass
class EvaluationResult:
    proposal: DishProposal
    vetoed: bool
    veto_reason: Optional[str]
    cuisine_blue_ocean: float
    trend_heat: float
    hawker_substitutability: float
    total_score: float
    passed: bool
    reasoning: str


@dataclass
class RoundResult:
    cuisine: str
    round_num: int
    proposals_generated: int
    passed_count: int
    rejected_count: int
    locked_total: int
    evaluations: list
    improvement_suggestions: str
    elapsed_seconds: float


@dataclass
class CuisineResult:
    cuisine: str
    total_rounds: int
    locked: list
    rounds_history: list


@dataclass
class FinalOutput:
    timestamp: str
    cuisines: dict
    total_elapsed_seconds: float


@dataclass
class RawCommodity:
    """Direct CSV row — weak structured data from Metabase export."""
    id: int
    name: str
    description: str


@dataclass
class ProcessedCommodity:
    """Enriched commodity with LLM-generated semantic tags."""
    id: int
    name: str
    description: str
    cuisine_type: str          # Chinese/Japanese/Korean/Thai/Singaporean-Malay/Mexican/Other
