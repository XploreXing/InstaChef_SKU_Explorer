"""Does recorded human feedback change the evaluator's next judgment?

The evaluator is given the rejections in evals/cases/feedback.yaml, as the
text the pipeline builds from them, and then scores dishes it has not seen.
A dish that a rejection reason applies to must be vetoed; an ordinary dish
must be scored as usual. Runs without the feedback show what the evaluator
does on its own.

Web search is switched off here, so a veto can only come from the feedback.
Each run is one real evaluator call.

Usage (from the repo root):
  python evals/run_feedback.py              # 5 runs with feedback, 2 without
  python evals/run_feedback.py --runs 10 --control-runs 3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yaml
from dotenv import load_dotenv

from agents.evaluator import NOT_SCORED_REASON, EvaluatorAgent
from utils.feedback_loader import build_evaluator_context

CASES_FILE = REPO / "evals" / "cases" / "feedback.yaml"


def load() -> tuple[str, list[dict], list[dict]]:
    doc = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))
    return doc["cuisine"], doc["rejections"], doc["cases"]


def evaluate_once(config: dict, cuisine: str, cases: list[dict], context: str) -> dict:
    """One evaluator call. Returns what it said about each case."""
    agent = EvaluatorAgent(config)
    chat = agent.client.chat
    # No tools: without web search, feedback is the only reason left to veto.
    agent.client.chat = lambda **kw: chat(**{k: v for k, v in kw.items() if k != "tools"})

    proposals = [{"id": i, "cuisine": cuisine, **case["dish"]} for i, case in enumerate(cases, start=1)]
    evaluations, _, _ = agent.evaluate(proposals=proposals, hitl_context=context)
    results = EvaluatorAgent.to_evaluation_results(evaluations, proposals)
    return {
        case["id"]: {
            "vetoed": r.vetoed,
            "missing": r.veto_reason == NOT_SCORED_REASON,
            "reason": r.veto_reason or "",
            "score": r.total_score,
        }
        for case, r in zip(cases, results)
    }


def summarize(cases: list[dict], rejections: list[dict], runs: list[dict]) -> dict:
    """Count, over the runs that returned a verdict for every dish, how often
    the evaluator vetoed what it should and left alone what it should."""
    precedents = [name for r in rejections
                  for name in (r.get("proposal_name_cn"), r.get("proposal_name")) if name]
    complete = [run for run in runs if not any(v["missing"] for v in run.values())]
    should_veto = [c["id"] for c in cases if c["expect"] == "veto"]
    should_keep = [c["id"] for c in cases if c["expect"] == "keep"]

    vetoed_right = sum(run[cid]["vetoed"] for run in complete for cid in should_veto)
    vetoed_wrong = sum(run[cid]["vetoed"] for run in complete for cid in should_keep)
    named = sum(
        any(p in run[cid]["reason"] for p in precedents)
        for run in complete for cid in should_veto if run[cid]["vetoed"]
    )
    fully_right = sum(
        all(run[cid]["vetoed"] for cid in should_veto) and not any(run[cid]["vetoed"] for cid in should_keep)
        for run in complete
    )
    return {
        "runs": len(runs),
        "complete": len(complete),
        "vetoed_right": [vetoed_right, len(should_veto) * len(complete)],
        "vetoed_wrong": [vetoed_wrong, len(should_keep) * len(complete)],
        "named_precedent": [named, vetoed_right],
        "fully_right": [fully_right, len(complete)],
    }


def print_report(name: str, cases: list[dict], runs: list[dict], summary: dict) -> None:
    print(f"\n━━ {name} ━━")
    print(f"{'dish':<36} {'expect':<8} verdict per run")
    for case in cases:
        marks = " ".join(
            " none" if run[case["id"]]["missing"] else " VETO" if run[case["id"]]["vetoed"]
            else f"{run[case['id']]['score']:5.1f}"
            for run in runs
        )
        print(f"{case['dish']['name'][:35]:<36} {case['expect']:<8} {marks}")

    right, chances = summary["vetoed_right"]
    wrong, keeps = summary["vetoed_wrong"]
    ok, complete = summary["fully_right"]
    print(f"\n  should be vetoed  : vetoed {right}/{chances}")
    print(f"  should be kept    : vetoed {wrong}/{keeps}")
    print(f"  runs fully right  : {ok}/{complete}")
    if right:
        named, _ = summary["named_precedent"]
        print(f"  vetoes that name the rejected dish they follow: {named}/{right}")
    if summary["complete"] < summary["runs"]:
        print(f"  not counted: {summary['runs'] - summary['complete']} runs where the evaluator "
              f"returned no score for some dish")
    for run in runs[:1]:
        for case in cases:
            v = run[case["id"]]
            if v["vetoed"] and not v["missing"]:
                print(f"  e.g. {case['dish']['name']}: {v['reason'][:110]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=5, help="runs with the feedback (default 5)")
    parser.add_argument("--control-runs", type=int, default=2, help="runs without it (default 2)")
    parser.add_argument("--preset", help="LLM preset id from config.yaml (default: config default)")
    parser.add_argument("--workers", type=int, default=3, help="evaluator calls in parallel")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    config = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    if args.preset:
        config["llm"]["selected_preset_id"] = args.preset
    cuisine, rejections, cases = load()
    context = build_evaluator_context(rejections, cuisine)

    jobs = [context] * args.runs + [""] * args.control_runs
    # The evaluator logs every loop to stdout; keep that out of the report.
    with contextlib.redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(lambda ctx: evaluate_once(config, cuisine, cases, ctx), jobs))

    print(f"{len(cases)} {cuisine} dishes, {len(rejections)} rejections given as feedback, "
          f"preset {config['llm'].get('selected_preset_id') or config['llm'].get('default_preset')}")
    print("feedback text given to the evaluator:" + context.replace("\n", "\n  | "))
    with_feedback, without = outcomes[: args.runs], outcomes[args.runs:]
    print_report("with the feedback", cases, with_feedback, summarize(cases, rejections, with_feedback))
    if without:
        print_report("without it (control)", cases, without, summarize(cases, rejections, without))
    return 0


if __name__ == "__main__":
    sys.exit(main())
