"""Measure the LLM evaluator on the fixed dishes in evals/cases/evaluator.yaml.

It answers two separate questions:

  agreement  Does the evaluator's verdict match the human label?
             Only dishes with a label count.
  stability  Does the same dish get the same verdict and score every run?
             Needs no labels: every dish counts.

Each run sends the dishes to the real evaluator, grouped by cuisine the way
the pipeline does, so it makes API calls and costs tokens. The evaluator's
web-search tool stays available, as in the pipeline.

No HITL context is passed. The labelled dishes are the recorded human
rejections, so telling the evaluator about them would hand it the answers.

Usage (from the repo root):
  python evals/run_evaluator.py                    # 3 runs, evaluator as configured
  python evals/run_evaluator.py --thinking both    # compare thinking on and off
  python evals/run_evaluator.py --runs 5 --preset siliconflow-deepseek-v4-flash
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
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
from agents.evaluator import EvaluatorAgent

CASES_FILE = REPO / "evals" / "cases" / "evaluator.yaml"
RESULTS_DIR = REPO / "evals" / "results"
SEARCH_CACHE_FILE = RESULTS_DIR / "search_cache.json"

# Rubric weights from prompts/evaluator_system.md, on raw 0-10 scores.
WEIGHTS = {"cuisine_blue_ocean": 4.0, "trend_heat": 3.5, "hawker_substitutability": 2.5}

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


def load_cases() -> tuple[dict, list[dict]]:
    doc = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))
    return doc["context"], doc["cases"]


def _verdict(ev: dict) -> dict:
    """What the evaluator said about one dish, plus the total recomputed in
    code from its own dimension scores."""
    scores = ev.get("scores") if isinstance(ev.get("scores"), dict) else {}
    raw = {dim: (scores.get(dim) or {}).get("raw") for dim in WEIGHTS}
    recomputed = None
    if all(isinstance(v, (int, float)) for v in raw.values()):
        recomputed = round(sum(raw[dim] * weight for dim, weight in WEIGHTS.items()), 1)
    return {
        "passed": bool(ev.get("passed")),
        "vetoed": bool(ev.get("vetoed")),
        "score": ev.get("total_score"),
        "raw": raw,
        "recomputed": recomputed,
        "veto_reason": ev.get("veto_reason"),
    }


def evaluate_cuisine(config: dict, cuisine: str, cases: list[dict], context: dict,
                     thinking: bool | None) -> dict:
    """One evaluator call for one cuisine. Returns {case id: verdict or None}
    plus the token and latency numbers of the call."""
    agent = EvaluatorAgent(config)
    if thinking is not None:
        # Override the evaluator's own choice for this comparison.
        chat = agent.client.chat
        agent.client.chat = lambda **kwargs: chat(**{**kwargs, "thinking": thinking})

    proposals = [{"id": i, **case["dish"]} for i, case in enumerate(cases, start=1)]
    started = time.time()
    evaluations, _, _ = agent.evaluate(
        proposals=proposals,
        cuisine_sku_count=context["sku_counts"].get(cuisine, 0),
        round_num=1,
        locked_count=0,
        remaining=10,
        pass_threshold=context["pass_threshold"],
    )
    elapsed = time.time() - started

    # Match by dish name, falling back to the id we sent. Position is not
    # trusted: the model may drop or reorder dishes.
    by_id = {ev.get("id"): ev for ev in evaluations if isinstance(ev, dict)}
    by_name = {_normalize(ev.get("name")): ev for ev in evaluations if isinstance(ev, dict)}
    verdicts = {}
    for proposal, case in zip(proposals, cases):
        ev = by_name.get(_normalize(proposal["name"])) or by_id.get(proposal["id"])
        verdicts[case["id"]] = None if ev is None else _verdict(ev)

    metrics = [m for m in agent.client.metrics if m.success]
    return {
        "verdicts": verdicts,
        "calls": len(agent.client.metrics),
        "prompt_tokens": sum(m.prompt_tokens for m in metrics),
        "completion_tokens": sum(m.completion_tokens for m in metrics),
        "seconds": elapsed,
    }


def run_mode(config: dict, context: dict, cases: list[dict], thinking: bool | None,
             runs: int, workers: int) -> dict:
    by_cuisine: dict[str, list[dict]] = {}
    for case in cases:
        by_cuisine.setdefault(case["dish"]["cuisine"], []).append(case)

    searches_before = dict(_search_counts)
    jobs = [(run, cuisine) for run in range(runs) for cuisine in by_cuisine]
    # The evaluator logs every loop to stdout; keep that out of the report.
    agent_log = io.StringIO()
    with contextlib.redirect_stdout(agent_log), ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(
            lambda job: evaluate_cuisine(config, job[1], by_cuisine[job[1]], context, thinking),
            jobs,
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


def summarize(cases: list[dict], result: dict, threshold: float) -> dict:
    per_case = result["per_case"]
    spreads, flips, missing, contradictions, miscalculated = [], [], [], 0, []
    agree = {"reject": [0, 0], "accept": [0, 0]}  # [verdicts that agree, verdicts given]
    always_agree = {"reject": [0, 0], "accept": [0, 0]}  # [dishes agreeing on every run, dishes]

    for case in cases:
        verdicts = per_case[case["id"]]
        given = [v for v in verdicts if v is not None]
        if len(given) < len(verdicts):
            missing.append(case["id"])
        scores = [v["score"] for v in given if isinstance(v["score"], (int, float))]
        if len(scores) >= 2:
            spreads.append((max(scores) - min(scores), case["id"]))
        if len({v["passed"] for v in given}) > 1:
            flips.append(case["id"])
        for v in given:
            by_score = (not v["vetoed"]) and isinstance(v["score"], (int, float)) and v["score"] >= threshold
            contradictions += v["passed"] != by_score
            # The model adds up its own dimension scores; check its arithmetic.
            if (not v["vetoed"] and v["recomputed"] is not None
                    and isinstance(v["score"], (int, float)) and abs(v["score"] - v["recomputed"]) > 1):
                miscalculated.append((case["id"], v["score"], v["recomputed"]))

        label = case.get("label")
        if label in agree and given:
            want_pass = label == "accept"
            hits = sum(v["passed"] == want_pass for v in given)
            agree[label][0] += hits
            agree[label][1] += len(given)
            always_agree[label][0] += hits == len(given)
            always_agree[label][1] += 1

    spread_values = [s for s, _ in spreads]
    return {
        "agree": agree,
        "always_agree": always_agree,
        "median_spread": statistics.median(spread_values) if spread_values else None,
        "max_spread": max(spreads) if spreads else None,
        "flips": flips,
        "missing": missing,
        "contradictions": contradictions,
        "miscalculated": miscalculated,
    }


def print_report(name: str, cases: list[dict], result: dict, summary: dict, runs: int) -> None:
    print(f"\n━━ {name} ━━")
    print(f"{'dish':<46} {'label':<7} scores per run          verdicts")
    for case in cases:
        verdicts = result["per_case"][case["id"]]
        scores = " ".join(f"{v['score']:>5}" if v else "    -" for v in verdicts)
        marks = " ".join(("veto" if v["vetoed"] else "PASS" if v["passed"] else "fail") if v else "none"
                         for v in verdicts)
        flag = "  <- flips" if case["id"] in summary["flips"] else ""
        print(f"{case['dish']['name'][:45]:<46} {case.get('label') or '-':<7} {scores:<22}  {marks}{flag}")

    totals = result["totals"]
    print(f"\n  cost       {totals['calls']} LLM calls for {result['batches']} batches, "
          f"{totals['prompt_tokens']:,} prompt + {totals['completion_tokens']:,} completion tokens, "
          f"{totals['seconds'] / max(result['batches'], 1):.0f}s per batch; "
          f"web searches: {totals['searches_live']} live, {totals['searches_cached']} replayed")
    for label, word in (("reject", "not passed"), ("accept", "passed")):
        hits, given = summary["agree"][label]
        dishes_ok, dishes = summary["always_agree"][label]
        if given:
            print(f"  agreement  human said {label}: evaluator {word} in {hits}/{given} verdicts; "
                  f"{dishes_ok}/{dishes} dishes on every run")
        else:
            print(f"  agreement  no dish is labelled {label} yet")
    if summary["median_spread"] is not None:
        worst, worst_id = summary["max_spread"]
        print(f"  stability  score spread across {runs} runs: median {summary['median_spread']:.1f}, "
              f"largest {worst:.1f} ({worst_id})")
    print(f"  stability  verdict changed between runs for {len(summary['flips'])}/{len(cases)} dishes")
    if summary["missing"]:
        print(f"  missing    no verdict in at least one run for {len(summary['missing'])} dishes: "
              f"{', '.join(summary['missing'][:4])}{' ...' if len(summary['missing']) > 4 else ''}")
    if summary["contradictions"]:
        print(f"  warning    {summary['contradictions']} verdicts where the model's `passed` flag "
              f"contradicts its own score and the threshold")
    if summary["miscalculated"]:
        examples = ", ".join(f"{cid} said {said} but its dimensions add up to {real}"
                             for cid, said, real in summary["miscalculated"][:3])
        print(f"  warning    {len(summary['miscalculated'])} verdicts where total_score is not the "
              f"weighted sum of its own dimension scores: {examples}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=3, help="times each dish is evaluated (default 3)")
    parser.add_argument("--thinking", choices=["default", "on", "off", "both"], default="default",
                        help="default = whatever the evaluator asks for; both = compare on and off")
    parser.add_argument("--preset", help="LLM preset id from config.yaml (default: config default)")
    parser.add_argument("--temperature", type=float,
                        help="override llm.evaluator_temperature (ignored by models while they think)")
    parser.add_argument("--threshold", type=float,
                        help="override the pass threshold, e.g. to replay a later round "
                             "after the orchestrator lowered it")
    parser.add_argument("--workers", type=int, default=4, help="cuisine batches run in parallel")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    config = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    if args.preset:
        config["llm"]["selected_preset_id"] = args.preset
    if args.temperature is not None:
        config["llm"]["evaluator_temperature"] = args.temperature
    context, cases = load_cases()
    if args.threshold is not None:
        context["pass_threshold"] = args.threshold
    _install_search_cache()

    modes = {"default": [("as configured", None)], "on": [("thinking on", True)],
             "off": [("thinking off", False)],
             "both": [("thinking on", True), ("thinking off", False)]}[args.thinking]
    labelled = sum(1 for c in cases if c.get("label"))
    print(f"{len(cases)} dishes ({labelled} with a human label), {args.runs} runs each, "
          f"preset {config['llm'].get('selected_preset_id') or config['llm'].get('default_preset')}, "
          f"temperature {config['llm']['evaluator_temperature']}, "
          f"pass threshold {context['pass_threshold']}")

    report = {"runs": args.runs, "context": context, "modes": {}}
    for name, thinking in modes:
        result = run_mode(config, context, cases, thinking, args.runs, args.workers)
        summary = summarize(cases, result, context["pass_threshold"])
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
