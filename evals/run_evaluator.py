"""Measure the LLM evaluator on the fixed dishes in evals/cases/evaluator.yaml.

The pipeline ranks the dishes of a cuisine by score and keeps the best ones,
so this asks two things about the scores:

  stability  Does a dish get the same score, and the same place in its
             cuisine's ranking, every run? Needs no labels.
  agreement  Does the ranking put dishes people accepted above dishes people
             rejected? Only labelled dishes count.

Each run sends the dishes to the real evaluator, grouped by cuisine the way
the pipeline does, so it makes API calls and costs tokens. The evaluator's
web-search tool stays available, as in the pipeline.

No HITL context is passed. The labelled dishes are the recorded human
rejections, so telling the evaluator about them would hand it the answers.

Usage (from the repo root):
  python evals/run_evaluator.py                    # 3 runs, evaluator as configured
  python evals/run_evaluator.py --thinking both    # compare thinking on and off
  python evals/run_evaluator.py --runs 5 --preset siliconflow-deepseek-v4-flash
  python evals/run_evaluator.py --thinking off --temperature 1.0
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yaml
from dotenv import load_dotenv

import agents.evaluator as evaluator_module
from agents.evaluator import NOT_SCORED_REASON, EvaluatorAgent

CASES_FILE = REPO / "evals" / "cases" / "evaluator.yaml"
RESULTS_DIR = REPO / "evals" / "results"
SEARCH_CACHE_FILE = RESULTS_DIR / "search_cache.json"

# The evaluator's web search costs quota and returns different results from
# day to day. Each distinct search is made once and replayed afterwards, so
# runs compare the model, not the search engine. Delete the cache file to
# refresh it.
_real_search = evaluator_module.search_web_for_eval
_search_lock = threading.Lock()
_search_cache: dict[str, str] = {}
_search_counts = {"live": 0, "cached": 0}


def _cached_search(**kwargs) -> str:
    key = f"{_normalize(kwargs.get('cuisine'))}|{_normalize(kwargs.get('menu'))}"
    with _search_lock:
        if key in _search_cache:
            _search_counts["cached"] += 1
            return _search_cache[key]
    result = _real_search(**kwargs)
    with _search_lock:
        _search_counts["live"] += 1
        if not result.startswith("⚠️"):  # do not freeze a failed search
            _search_cache[key] = result
    return result


def _install_search_cache() -> None:
    """Route the evaluator's search tool through the replay cache. Done when a
    run starts, not on import, so importing this module has no side effects."""
    if SEARCH_CACHE_FILE.exists():
        _search_cache.update(json.loads(SEARCH_CACHE_FILE.read_text(encoding="utf-8")))
    evaluator_module.search_web_for_eval = _cached_search


def _normalize(name: str) -> str:
    return " ".join((name or "").lower().split())


def load_cases() -> list[dict]:
    return yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))["cases"]


def evaluate_cuisine(config: dict, cases: list[dict], thinking: bool | None) -> dict:
    """One evaluator call for the dishes of one cuisine. Returns what it said
    about each case, plus the token and latency numbers of the call."""
    agent = EvaluatorAgent(config)
    if thinking is not None:
        # Override the evaluator's own choice for this comparison.
        chat = agent.client.chat
        agent.client.chat = lambda **kwargs: chat(**{**kwargs, "thinking": thinking})

    proposals = [{"id": i, **case["dish"]} for i, case in enumerate(cases, start=1)]
    started = time.time()
    evaluations, _, _ = agent.evaluate(proposals=proposals)
    elapsed = time.time() - started

    # The same conversion the pipeline uses: matched by name, total computed in code.
    results = EvaluatorAgent.to_evaluation_results(evaluations, proposals)
    verdicts = {
        case["id"]: {
            "scored": not r.vetoed,
            "missing": r.veto_reason == NOT_SCORED_REASON,
            "score": r.total_score,
            "trend": r.trend_heat,
            "hawker": r.hawker_substitutability,
            "veto_reason": r.veto_reason,
        }
        for case, r in zip(cases, results)
    }

    metrics = [m for m in agent.client.metrics if m.success]
    return {
        "verdicts": verdicts,
        "calls": len(agent.client.metrics),
        "prompt_tokens": sum(m.prompt_tokens for m in metrics),
        "completion_tokens": sum(m.completion_tokens for m in metrics),
        "seconds": elapsed,
    }


def _by_cuisine(cases: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(case["dish"]["cuisine"], []).append(case)
    return groups


def run_mode(config: dict, cases: list[dict], thinking: bool | None,
             runs: int, workers: int) -> dict:
    groups = _by_cuisine(cases)
    searches_before = dict(_search_counts)
    jobs = [(run, cuisine) for run in range(runs) for cuisine in groups]
    # The evaluator logs every loop to stdout; keep that out of the report.
    agent_log = io.StringIO()
    with contextlib.redirect_stdout(agent_log), ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(
            lambda job: evaluate_cuisine(config, groups[job[1]], thinking), jobs,
        ))

    per_case: dict[str, list] = {case["id"]: [None] * runs for case in cases}
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    for (run, _), outcome in zip(jobs, outcomes):
        for case_id, verdict in outcome["verdicts"].items():
            per_case[case_id][run] = verdict
        for key in totals:
            totals[key] += outcome[key]
    for kind in _search_counts:
        totals[f"searches_{kind}"] = _search_counts[kind] - searches_before[kind]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_CACHE_FILE.write_text(
        json.dumps(_search_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"per_case": per_case, "totals": totals, "batches": len(jobs),
            "agent_log": agent_log.getvalue()}


def top_half(cases: list[dict], per_case: dict, run: int) -> set[str]:
    """The dishes of one cuisine that the pipeline would keep in `run` if the
    quota were half the batch (rounded up). A vetoed dish is never kept."""
    scored = [c["id"] for c in cases if per_case[c["id"]][run]["scored"]]
    scored.sort(key=lambda cid: per_case[cid][run]["score"], reverse=True)
    return set(scored[: math.ceil(len(cases) / 2)])


def summarize(cases: list[dict], result: dict, runs: int) -> dict:
    per_case = result["per_case"]
    kept = {case["id"]: [] for case in cases}       # per run: was the dish in its cuisine's top half?
    pairs_right, pairs_total = 0, 0
    for group in _by_cuisine(cases).values():
        accepted = [c["id"] for c in group if c.get("label") == "accept"]
        rejected = [c["id"] for c in group if c.get("label") == "reject"]
        for run in range(runs):
            selected = top_half(group, per_case, run)
            for case in group:
                kept[case["id"]].append(case["id"] in selected)
            # Ranking agreement: an accepted dish should outscore a rejected one.
            for a in accepted:
                for r in rejected:
                    va, vr = per_case[a][run], per_case[r][run]
                    pairs_total += 1
                    pairs_right += va["scored"] and (not vr["scored"] or va["score"] > vr["score"])

    spreads, veto_flips = [], []
    for case in cases:
        verdicts = per_case[case["id"]]
        scores = [v["score"] for v in verdicts if v["scored"]]
        if len(scores) >= 2:
            spreads.append((round(max(scores) - min(scores), 1), case["id"]))
        if 0 < len(scores) < len(verdicts):
            veto_flips.append(case["id"])

    rejected_ids = [c["id"] for c in cases if c.get("label") == "reject"]
    spread_values = [s for s, _ in spreads]
    return {
        "kept": kept,
        "rank_flips": [cid for cid, flags in kept.items() if len(set(flags)) > 1],
        "veto_flips": veto_flips,
        "median_spread": statistics.median(spread_values) if spread_values else None,
        "max_spread": max(spreads) if spreads else None,
        "missing": sum(v["missing"] for vs in per_case.values() for v in vs),
        "pairs": [pairs_right, pairs_total],
        "rejected_kept": [sum(sum(kept[cid]) for cid in rejected_ids), len(rejected_ids) * runs],
        "rejected_always_kept": sum(all(kept[cid]) for cid in rejected_ids),
    }


def print_report(name: str, cases: list[dict], result: dict, summary: dict, runs: int) -> None:
    print(f"\n━━ {name} ━━")
    print(f"{'dish':<42} {'cuisine':<12} {'label':<7} scores per run       kept (top half of its cuisine)")
    for case in cases:
        verdicts = result["per_case"][case["id"]]
        scores = " ".join(f"{v['score']:>5}" if v["scored"] else " none" if v["missing"] else " veto"
                          for v in verdicts)
        marks = " ".join("yes" if k else " no" for k in summary["kept"][case["id"]])
        flag = "  <- changes" if case["id"] in summary["rank_flips"] else ""
        print(f"{case['dish']['name'][:41]:<42} {case['dish']['cuisine'][:11]:<12} "
              f"{case.get('label') or '-':<7} {scores:<20}  {marks}{flag}")

    totals = result["totals"]
    print(f"\n  cost       {totals['calls']} LLM calls for {result['batches']} batches, "
          f"{totals['prompt_tokens']:,} prompt + {totals['completion_tokens']:,} completion tokens, "
          f"{totals['seconds'] / max(result['batches'], 1):.0f}s per batch; "
          f"web searches: {totals['searches_live']} live, {totals['searches_cached']} replayed")
    if summary["median_spread"] is not None:
        worst, worst_id = summary["max_spread"]
        print(f"  stability  score spread across {runs} runs: median {summary['median_spread']:.1f}, "
              f"largest {worst:.1f} ({worst_id})")
    print(f"  stability  kept in some runs but not others: {len(summary['rank_flips'])}/{len(cases)} dishes; "
          f"vetoed in some runs but not others: {len(summary['veto_flips'])}")
    right, total = summary["pairs"]
    if total:
        print(f"  agreement  an accepted dish outscores a rejected one of its cuisine in "
              f"{right}/{total} comparisons")
    else:
        print("  agreement  no cuisine has both an accepted and a rejected dish yet, "
              "so the ranking cannot be checked against people")
    kept, chances = summary["rejected_kept"]
    if chances:
        print(f"  agreement  dishes people rejected were kept {kept}/{chances} times; "
              f"{summary['rejected_always_kept']} of them on every run")
    if summary["missing"]:
        print(f"  missing    the evaluator returned no score {summary['missing']} times")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=3, help="times each dish is evaluated (default 3)")
    parser.add_argument("--thinking", choices=["default", "on", "off", "both"], default="default",
                        help="default = whatever the evaluator asks for; both = compare on and off")
    parser.add_argument("--preset", help="LLM preset id from config.yaml (default: config default)")
    parser.add_argument("--temperature", type=float,
                        help="override llm.evaluator_temperature (ignored by models while they think)")
    parser.add_argument("--workers", type=int, default=4, help="cuisine batches run in parallel")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    config = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    if args.preset:
        config["llm"]["selected_preset_id"] = args.preset
    if args.temperature is not None:
        config["llm"]["evaluator_temperature"] = args.temperature
    cases = load_cases()
    _install_search_cache()

    modes = {"default": [("as configured", None)], "on": [("thinking on", True)],
             "off": [("thinking off", False)],
             "both": [("thinking on", True), ("thinking off", False)]}[args.thinking]
    labelled = sum(1 for c in cases if c.get("label"))
    print(f"{len(cases)} dishes ({labelled} with a human label), {args.runs} runs each, "
          f"preset {config['llm'].get('selected_preset_id') or config['llm'].get('default_preset')}, "
          f"temperature {config['llm']['evaluator_temperature']}")

    report = {"runs": args.runs, "modes": {}}
    for name, thinking in modes:
        result = run_mode(config, cases, thinking, args.runs, args.workers)
        summary = summarize(cases, result, args.runs)
        print_report(name, cases, result, summary, args.runs)
        report["modes"][name] = {"result": result, "summary": summary}

    # Kept out of git (.gitignore): run artefacts, including the agent's own log.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"evaluator-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nraw results: {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
