import json
import os
import sys
import time
import threading
from dataclasses import asdict
from pathlib import Path

# Load .env before any other imports that read environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from models import RejectionFeedback

REJECTION_REASONS = {
    "cold_food":     ("冷食/不适合60-70°C热柜", "guard_kill"),
    "non_halal":     ("不清真/含猪肉酒精",       "guard_haram"),
    "fried":         ("油炸/不健康",             "evaluator_fried"),
    "duplicate":     ("已有类似SKU/重复",        "evaluator_duplicate"),
    "bad_taste":     ("口味不适合新加坡市场",     "evaluator_trend"),
    "bad_name":      ("菜名不够吸引人",          "generator_naming"),
    "too_complex":   ("做法太复杂/不适合自动化", "evaluator_complexity"),
    "cost_high":     ("原料成本过高",             "generator_price"),
    "other":         ("其他（自定义）",           "manual_review"),
}

st.set_page_config(
    page_title="InstaChef SKU Explorer",
    page_icon="🔍",
    layout="wide",
)

# File-based progress tracking — the ONLY reliable way to bridge
# background threads → Streamlit UI across st.rerun() cycles.
# Neither st.session_state nor module-level vars survive Streamlit's
# script-reload semantics; files written by the bg thread and read by
# the main thread work 100% of the time.
_PROGRESS_FILE = Path(__file__).resolve().parent / "data" / "runtime" / "pipeline_progress.json"
_OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "runtime" / "pipeline_output.json"


def _read_progress() -> dict | None:
    if not _PROGRESS_FILE.exists():
        return None
    try:
        with open(_PROGRESS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_progress(state: dict):
    """Atomic write: write to temp file first, then rename. Prevents
    the UI from reading a half-written JSON file."""
    _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(_PROGRESS_FILE)  # atomic on macOS/Linux


def _clear_progress():
    try:
        _PROGRESS_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        _OUTPUT_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _save_output(output) -> Path:
    """Serialize FinalOutput to JSON file. Returns the file path."""
    data = asdict(output)
    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return _OUTPUT_FILE


def _save_executive_summary(output, log_dir: Path) -> Path | None:
    """Save ExecutiveSummary to a standalone JSON file."""
    if output.executive_summary is None:
        return None
    from dataclasses import asdict
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"executive_summary_{ts}.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(output.executive_summary), f, ensure_ascii=False, indent=2)
    return path


def _save_lineage_log(output, log_dir: Path) -> Path | None:
    """Save search-ref mapping and proposal reference validation to a JSON file.
    Lets developers verify Generator claims against Tavily sources."""
    from datetime import datetime

    entries: list[dict] = []
    for cuisine, cr in output.cuisines.items():
        for rr in cr.rounds_history:
            if rr.ref_map or rr.lineage_results:
                entries.append({
                    "cuisine": cuisine,
                    "round": rr.round_num,
                    "search_refs": {
                        tag: url for tag, url in rr.ref_map.items()
                    },
                    "proposals": rr.lineage_results,
                })

    if not entries:
        return None

    # Quick stats (dual-evidence)
    total = sum(len(e["proposals"]) for e in entries)
    validated = sum(
        1 for e in entries for p in e["proposals"] if p.get("validated")
    )
    hallucinated = total - validated
    dish_ok = sum(
        1 for e in entries for p in e["proposals"]
        if p.get("validated") and p.get("dish_name_matched")
    )
    trend_ok = sum(
        1 for e in entries for p in e["proposals"]
        if p.get("validated") and p.get("trend_matched")
    )
    menu_count = sum(
        1 for e in entries for p in e["proposals"]
        if p.get("evidence_level") == "menu"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"lineage_{ts}.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "timestamp": output.timestamp,
            "stats": {
                "total": total,
                "tag_validated": validated,
                "hallucinated": hallucinated,
                "dish_name_matched": dish_ok,
                "trend_matched": trend_ok,
                "menu_evidence": menu_count,
            },
            "entries": entries,
        }, f, ensure_ascii=False, indent=2)
    return path


def _save_tracing_log(output, log_dir: Path) -> Path | None:
    """Save per-stage performance traces to a standalone JSON file."""
    from collections import defaultdict
    from datetime import datetime

    traces: list[dict] = []
    for cuisine, cr in output.cuisines.items():
        for rr in cr.rounds_history:
            stages = []
            for t in rr.stage_traces:
                stages.append({
                    "stage": t.stage,
                    "elapsed_ms": t.elapsed_ms,
                    "elapsed_s": round(t.elapsed_ms / 1000, 1),
                    "model_name": t.model_name,
                    "input_size_chars": t.input_size_chars,
                    "output_size_chars": t.output_size_chars,
                })
            if stages:
                traces.append({
                    "cuisine": cuisine,
                    "round": rr.round_num,
                    "total_round_s": round(rr.elapsed_seconds, 1),
                    "stages": stages,
                })

    if not traces:
        return None

    stage_totals: dict[str, list[float]] = defaultdict(list)
    for t in traces:
        for s in t["stages"]:
            stage_totals[s["stage"]].append(s["elapsed_ms"])

    summary = {}
    for stage, times in stage_totals.items():
        avg_s = sum(times) / len(times) / 1000
        summary[f"{stage}_avg_s"] = round(avg_s, 1)
        summary[f"{stage}_count"] = len(times)
        summary[f"{stage}_pct"] = round(100 * sum(times) / (output.total_elapsed_seconds * 1000), 1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"tracing_{ts}.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "timestamp": output.timestamp,
            "total_elapsed_s": round(output.total_elapsed_seconds, 1),
            "traces": traces,
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)
    return path


def _save_evaluation_log(output, log_dir: Path):
    """Save per-proposal evaluation details with reasons to a timestamped JSON file."""
    from datetime import datetime

    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"evaluation_log_{ts}.json"

    records = []
    for cuisine, cr in output.cuisines.items():
        for rr in cr.rounds_history:
            for ev in rr.evaluations:
                records.append({
                    "cuisine": cuisine,
                    "round": rr.round_num,
                    "name": ev.proposal.name,
                    "name_cn": ev.proposal.name_cn,
                    "description": ev.proposal.description,
                    "description_cn": ev.proposal.description_cn,
                    "vetoed": ev.vetoed,
                    "veto_reason": ev.veto_reason,
                    "cuisine_blue_ocean": ev.cuisine_blue_ocean,
                    "trend_heat": ev.trend_heat,
                    "hawker_substitutability": ev.hawker_substitutability,
                    "total_score": ev.total_score,
                    "passed": ev.passed,
                    "reasoning": ev.reasoning,
                })

    with open(log_path, "w") as f:
        json.dump({"timestamp": ts, "evaluations": records}, f, ensure_ascii=False, indent=2)

    print(f"📝 Evaluation log saved: {log_path}", flush=True)
    return log_path


def _load_output():
    """Load FinalOutput from JSON file. Returns None if not available."""
    if not _OUTPUT_FILE.exists():
        return None
    from models import FinalOutput, CuisineResult, RoundResult, EvaluationResult, DishProposal

    with open(_OUTPUT_FILE, "r") as f:
        data = json.load(f)

    cuisines = {}
    for name, cr_data in data.get("cuisines", {}).items():
        rounds = []
        for rr_data in cr_data.get("rounds_history", []):
            evals = []
            for e_data in rr_data.get("evaluations", []):
                p_data = e_data.get("proposal", {})
                prop = DishProposal(
                    id=p_data.get("id", 0),
                    name=p_data.get("name", ""),
                    name_cn=p_data.get("name_cn", ""),
                    cuisine=p_data.get("cuisine", ""),
                    description=p_data.get("description", ""),
                    description_cn=p_data.get("description_cn", ""),
                    price_sgd=p_data.get("price_sgd", 0),
                    differentiation=p_data.get("differentiation", ""),
                    trend_source=p_data.get("trend_source", ""),
                )
                evals.append(EvaluationResult(
                    proposal=prop,
                    vetoed=e_data.get("vetoed", False),
                    veto_reason=e_data.get("veto_reason"),
                    cuisine_blue_ocean=e_data.get("cuisine_blue_ocean", 0),
                    trend_heat=e_data.get("trend_heat", 0),
                    hawker_substitutability=e_data.get("hawker_substitutability", 0),
                    total_score=e_data.get("total_score", 0),
                    passed=e_data.get("passed", False),
                    reasoning=e_data.get("reasoning", ""),
                ))
            rounds.append(RoundResult(
                cuisine=rr_data.get("cuisine", ""),
                round_num=rr_data.get("round_num", 0),
                proposals_generated=rr_data.get("proposals_generated", 0),
                passed_count=rr_data.get("passed_count", 0),
                rejected_count=rr_data.get("rejected_count", 0),
                locked_total=rr_data.get("locked_total", 0),
                evaluations=evals,
                improvement_suggestions=rr_data.get("improvement_suggestions", ""),
                elapsed_seconds=rr_data.get("elapsed_seconds", 0),
            ))

        locked_evals = []
        for e_data in cr_data.get("locked", []):
            p_data = e_data.get("proposal", {})
            prop = DishProposal(
                id=p_data.get("id", 0),
                name=p_data.get("name", ""),
                name_cn=p_data.get("name_cn", ""),
                cuisine=p_data.get("cuisine", ""),
                description=p_data.get("description", ""),
                description_cn=p_data.get("description_cn", ""),
                price_sgd=p_data.get("price_sgd", 0),
                differentiation=p_data.get("differentiation", ""),
                trend_source=p_data.get("trend_source", ""),
                source_urls=p_data.get("source_urls", p_data.get("source_refs", [])),
            )
            locked_evals.append(EvaluationResult(
                proposal=prop,
                vetoed=e_data.get("vetoed", False),
                veto_reason=e_data.get("veto_reason"),
                cuisine_blue_ocean=e_data.get("cuisine_blue_ocean", 0),
                trend_heat=e_data.get("trend_heat", 0),
                hawker_substitutability=e_data.get("hawker_substitutability", 0),
                total_score=e_data.get("total_score", 0),
                passed=e_data.get("passed", False),
                reasoning=e_data.get("reasoning", ""),
            ))

        cuisines[name] = CuisineResult(
            cuisine=name,
            total_rounds=cr_data.get("total_rounds", 0),
            locked=locked_evals,
            rounds_history=rounds,
        )

    return FinalOutput(
        timestamp=data.get("timestamp", ""),
        cuisines=cuisines,
        total_elapsed_seconds=data.get("total_elapsed_seconds", 0),
    )


def _save_feedback(rejections: list[RejectionFeedback], log_dir: Path) -> Path:
    """Save rejection feedback to a timestamped JSON file."""
    from dataclasses import asdict
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"rejections_{ts}.json"
    with open(path, "w") as f:
        json.dump({
            "timestamp": ts,
            "count": len(rejections),
            "rejections": [asdict(r) for r in rejections],
        }, f, ensure_ascii=False, indent=2)
    return path


def _load_all_rejections(log_dir: Path) -> list[dict]:
    """Load all rejection feedback from a directory."""
    all_rejections: list[dict] = []
    if not log_dir.exists():
        return all_rejections
    for f in sorted(log_dir.glob("rejections_*.json")):
        try:
            data = json.loads(f.read_text())
            all_rejections.extend(data.get("rejections", []))
        except (json.JSONDecodeError, OSError):
            pass
    return all_rejections


@st.cache_resource
def _get_circuit_breaker(cooldown_seconds: float = 60.0):
    """进程级 CircuitBreaker 单例。

    缓存为 @st.cache_resource 单例：熔断冷却状态（_cooldowns dict）跨
    Streamlit rerun 保持。若 rerun new 新实例，_cooldowns 每次归空，刚熔断
    的 preset 会被立刻重试，等于没熔断——参见 utils/llm_circuit_breaker.py
    顶部 STREAMLIT PERSISTENCE 注释。
    """
    from utils.llm_circuit_breaker import CircuitBreaker
    return CircuitBreaker(cooldown_seconds=cooldown_seconds)


def init_session():
    defaults = {
        "output": None,
        "adopted": set(),
        "rejected": set(),
        "pending_rejections": [],
        "rejection_reasons_map": {},  # {dish_name: (reason_code, custom_note)}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_control_panel():
    with st.sidebar:
        st.header("⚙️ 控制区")

        cuisines = st.multiselect(
            "选择菜系",
            options=["Chinese", "Japanese", "Korean", "Thai", "Singaporean/Malay", "Mexican"],
            default=["Mexican"],
        )

        with st.expander("⚡ 高级参数"):
            st.slider("目标数/菜系", 1, 20, 10, key="target")
            st.slider("及格线", 50, 100, 80, key="threshold")
            st.slider("最大轮数", 1, 5, 3, key="max_rounds")

        with st.expander("🔑 模型配置"):
            # Preset dropdown: closed set of company-approved models.
            # Missing-API-key presets are filtered out at render time (mode A:
            # keys come from .env, not user input). Internal-tool form:
            # IT configures keys, users just pick a model.
            import yaml as _yaml
            with open("config.yaml") as _f:
                _cfg = _yaml.safe_load(_f)
            _all_presets = _cfg["llm"].get("presets", [])
            _available = [p for p in _all_presets if os.getenv(p["api_key_env"])]
            if not _available:
                st.error(
                    "没有任何 preset 配置了 API Key。请在 .env 中设置 "
                    "DEEPSEEK_API_KEY / SILICONFLOW_API_KEY / GEMINI_API_KEY 至少一个。"
                )
                st.stop()
            _preset_ids = [p["id"] for p in _available]
            _default_idx = 0
            _default_preset = _cfg["llm"].get("default_preset", _preset_ids[0])
            if _default_preset in _preset_ids:
                _default_idx = _preset_ids.index(_default_preset)
            st.selectbox(
                "模型预设",
                options=_preset_ids,
                index=_default_idx,
                format_func=lambda i: next(p["label"] for p in _available if p["id"] == i),
                key="selected_preset_id",
                help="API Key 全部来自 .env 环境变量；未配置 key 的预设不会出现在列表中",
            )

        progress = _read_progress()
        running = progress is not None and not progress.get("done", False)

        col_start, col_stop = st.columns([3, 1])

        with col_start:
            start_clicked = st.button(
                "🚀 启动搜索" if not running else "⏳ 搜索中...",
                type="primary",
                use_container_width=True,
                disabled=running,
            )

        with col_stop:
            stop_clicked = st.button(
                "⏹️ 停止",
                use_container_width=True,
                disabled=not running,
                type="secondary",
            )

        if stop_clicked:
            progress = _read_progress()
            if progress:
                progress["stop_requested"] = True
                _write_progress(progress)
            st.toast("⏹️ 停止请求已发送，等待当前 LLM 调用完成后生效")
            st.rerun()

        if start_clicked:
            st.session_state.output = None
            st.session_state.adopted = set()
            st.session_state.rejected = set()
            st.session_state._run_id = time.time()
            _clear_progress()

            # Capture sidebar params before starting thread
            # (st.session_state is not thread-safe)
            target = st.session_state.get("target", 10)
            threshold = st.session_state.get("threshold", 80)
            max_rounds = st.session_state.get("max_rounds", 3)
            selected_preset_id = st.session_state.get("selected_preset_id", "")

            state = {
                "done": False,
                "messages": [],
                "error": None,
                "cuisines": {},        # NEW: per-cuisine progress
                "total_locked": 0,
                "target": target,
                "stop_requested": False,
                "run_id": st.session_state.get("_run_id", 0),
            }
            _write_progress(state)

            def run_pipeline():
                _progess_lock=threading.Lock()
                def _write_progress_threadsafe(state):
                    with _progess_lock:
                        _write_progress(state)
                        
                def log(msg):
                    state["messages"].append(msg)
                    #原先这里是无锁调用 导致race condition（_write_progress(state))
                    _write_progress_threadsafe(state)
                    print(msg, flush=True)

                def update_state(cuisine, round_num, phase, meta):
                    if "cuisines" not in state:
                        state["cuisines"]={}
                    state["cuisines"][cuisine]={
                        "phase":phase,
                        "locked_count":meta.get("locked_count", 0),
                        "remaining":meta.get("remaining",0)
                    }
                    state["total_locked"] = sum(c.get("locked_count", 0) 
                                                for c in state["cuisines"].values()
                    )
                    _write_progress_threadsafe(state)

                try:
                    from utils.observability import ObservabilityLogger

                    obs = ObservabilityLogger(
                        session_id=str(st.session_state.get("_run_id", "")),
                    )
                    log(f"📊 trace_id: {obs.trace_id}")

                    orch = Orchestrator("config.yaml")
                    orch.obs = obs
                    # Override config with sidebar parameters
                    orch.config["orchestrator"]["target_per_cuisine"] = target
                    orch.config["orchestrator"]["pass_threshold"] = threshold
                    orch.config["orchestrator"]["max_rounds_per_cuisine"] = max_rounds

                    # Inject selected preset + shared CircuitBreaker into orchestrator.
                    # The breaker is a process-wide singleton via @st.cache_resource
                    # so cooldown state persists across Streamlit reruns; the
                    # orchestrator/agents/router are rebuilt each rerun (to reflect
                    # the latest preset choice) but share this one breaker.
                    orch.config["llm"]["selected_preset_id"] = selected_preset_id
                    orch.breaker = _get_circuit_breaker(orch.config["llm"].get("cooldown_seconds", 60))

                    log("🔬 正在用 AI 给 137 条商品做菜系分类...")
                    orch.load_skus()
                    counts = orch.sku_loader.get_cuisine_counts()
                    log(
                        f"✅ 分类完成: "
                        + ", ".join(f"{k}{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
                    )

                    log("🔍 开始搜索 + 生成 + 评估循环...")

                    # Wire stop-check: thread reads progress file to detect stop button click
                    orch.stop_check = lambda: _read_progress().get("stop_requested", False) if _read_progress() else False

                    orch.state_callbacks["on_state_change"].append(
                        lambda c, r, p, m: (
                            update_state(c, r, p, m),
                            log(
                                f"🔍 **{c}** · Round {r} · `{p}` · "
                                f"已锁定 {m.get('locked_count', 0)}/"
                                f"{m.get('remaining', 10) + m.get('locked_count', 0)}"
                            ),
                        )
                    )
                    def on_round_complete(res):
                        state["locked_count"] = min(res.locked_total, target)
                        _write_progress(state)
                        log(
                            f"✅ **{res.cuisine}** Round {res.round_num}: "
                            f"生成 {res.proposals_generated} → "
                            f"✅{res.passed_count} / ❌{res.rejected_count} → "
                            f"锁定 {min(res.locked_total, target)}/{target} "
                            f"({res.elapsed_seconds:.1f}s)"
                        )

                    orch.state_callbacks["on_round_complete"].append(on_round_complete)

                    output = orch.run(cuisines)
                    was_stopped = _read_progress().get("stop_requested", False) if _read_progress() else False

                    if was_stopped:
                        obs.flush()
                        log("⏹️ 用户中断。已保存当前进度。")
                    else:
                        _save_output(output)
                        log_path = _save_evaluation_log(output, Path("data/logs"))
                        log(f"📝 评估日志已保存: {log_path.name}")
                        summary_path = _save_executive_summary(output, Path("data/logs"))
                        if summary_path:
                            log(f"📊 高管摘要已保存: {summary_path.name}")
                        trace_path = _save_tracing_log(output, Path("data/logs"))
                        if trace_path:
                            log(f"⏱ 追踪日志已保存: {trace_path.name}")
                        lineage_path = _save_lineage_log(output, Path("data/logs"))
                        if lineage_path:
                            log(f"🔗 数据谱系已保存: {lineage_path.name}")
                        obs.flush()
                        log(f"📊 事件日志已保存: {obs.log_path.name} ({obs.event_count} events, trace_id={obs.trace_id})")
                        log("🎉 搜索完成！")
                except Exception as e:
                    state["error"] = str(e)
                    _write_progress(state)
                    print(f"❌ Pipeline error: {e}", flush=True)
                finally:
                    state["done"] = True
                    _write_progress(state)

            thread = threading.Thread(target=run_pipeline, daemon=True)
            thread.start()
            st.rerun()

        st.divider()

        history_files = sorted(
            Path("output").glob("sku_recommendations_*.json"),
            reverse=True,
        )
        if history_files:
            st.subheader("📁 历史记录")
            for f in history_files[:10]:
                st.caption(f"📄 {f.stem}")


def render_progress():
    progress = _read_progress()

    # Ignore stale progress from a previous run
    if progress is not None:
        current_run_id = st.session_state.get("_run_id", 0)
        progress_run_id = progress.get("run_id", 0)
        if current_run_id and progress_run_id and abs(current_run_id - progress_run_id) > 5:
            progress = None

    if progress is None:
        st.info("点击侧边栏 🚀 启动搜索 开始探索")
        return

    if progress.get("error"):
        st.error(f"运行出错: {progress['error']}")
        return

    if progress.get("done") and _OUTPUT_FILE.exists():
        st.success("✅ 搜索完成！切换到「📋 推荐结果」查看")
        for msg in progress.get("messages", []):
            st.write(msg)
        return

    if progress.get("done"):
        st.success("✅ 搜索完成！")
        for msg in progress.get("messages", []):
            st.write(msg)
        return

    # Still running — show progress bar + messages
    if progress.get("stop_requested"):
        st.warning("⏹️ 已请求停止 — 等待当前 LLM 调用完成后将自动终止")

    target_val = progress.get("target", 10)
    total_locked = progress.get("total_locked", 0)

    # Per-cuisine progress cards
    cuisines_state = progress.get("cuisines", {})
    if cuisines_state:
        # Overall progress
        if target_val > 0:
            overall_pct = min(total_locked / (target_val * len(cuisines_state)), 1.0)
            st.progress(overall_pct, text=f"总进度: 已锁定 {total_locked} / 目标 {target_val}×{len(cuisines_state)}菜系")

        # Per-cuisine cards side by side
        cols = st.columns(min(len(cuisines_state), 3))
        for i, (c_name, c_state) in enumerate(cuisines_state.items()):
            col = cols[i % len(cols)]
            with col:
                phase = c_state.get("phase", "")
                locked_count = c_state.get("locked_count", 0)
                remaining = c_state.get("remaining", 0)

                phase_emoji = {
                    "generating": "🤖",
                    "evaluating": "📊",
                    "judging": "⚖️",
                }.get(phase, "⏳")

                st.markdown(f"**{phase_emoji} {c_name}** — `{phase}`")
                if target_val > 0:
                    st.progress(
                        min(locked_count / target_val, 1.0),
                        text=f"锁定 {locked_count}/{target_val}",
                    )
                else:
                    st.caption(f"锁定 {locked_count}")
    else:
        # Fallback: no per-cuisine data yet (early startup)
        st.info("⏳ 正在初始化...")

    for msg in progress.get("messages", []):
        st.write(msg)


def render_results():
    if not st.session_state.output:
        st.info("还没有搜索结果，请先在侧边栏启动搜索")
        return

    output = st.session_state.output

    cuisine_filter = st.selectbox(
        "筛选菜系",
        options=["全部"] + list(output.cuisines.keys()),
    )

    all_evals = []
    for cuisine, cr in output.cuisines.items():
        if cuisine_filter != "全部" and cuisine != cuisine_filter:
            continue
        all_evals.extend(cr.locked)

    all_evals.sort(key=lambda e: e.total_score, reverse=True)

    st.subheader(f"📋 推荐结果（{len(all_evals)} 个）")

    rows = []
    for i, e in enumerate(all_evals):
        name = e.proposal.name
        # Combine CN+EN for display
        display_name = name
        if e.proposal.name_cn:
            display_name = f"{e.proposal.name_cn}\n{name}"
        display_desc = e.proposal.description[:80]
        if e.proposal.description_cn:
            display_desc = f"{e.proposal.description_cn[:80]}"
        rows.append({
            "采纳": name in st.session_state.adopted,
            "排名": i + 1,
            "菜品": display_name,
            "菜系": e.proposal.cuisine,
            "蓝海(40)": f"{e.cuisine_blue_ocean * 4:.0f}",
            "趋势(35)": f"{e.trend_heat * 3.5:.0f}",
            "替代(25)": f"{e.hawker_substitutability * 2.5:.0f}",
            "总分": f"{e.total_score:.1f}",
            "描述": display_desc,
        })

    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df,
        column_config={
            "采纳": st.column_config.CheckboxColumn("采纳"),
        },
        hide_index=True,
        use_container_width=True,
        key="results_table",
    )

    for idx, row in edited.iterrows():
        real_name = all_evals[idx].proposal.name
        if row["采纳"]:
            st.session_state.adopted.add(real_name)
            st.session_state.rejected.discard(real_name)
            st.session_state.rejection_reasons_map.pop(real_name, None)
        else:
            st.session_state.rejected.add(real_name)
            st.session_state.adopted.discard(real_name)

    # Rejection reason selectors for rejected items
    rejected_items = [
        (idx, e) for idx, e in enumerate(all_evals)
        if e.proposal.name in st.session_state.rejected
    ]
    if rejected_items:
        with st.expander("📝 拒绝原因（选择后将在导出页提交）", expanded=len(rejected_items) <= 3):
            for idx, e in rejected_items:
                name = e.proposal.name
                name_cn = e.proposal.name_cn or name
                current = st.session_state.rejection_reasons_map.get(name, ("", ""))
                col1, col2 = st.columns([1, 1])
                with col1:
                    reason = st.selectbox(
                        f"拒绝原因: {name_cn}",
                        options=["（未选择）"] + [label for _, (label, _) in REJECTION_REASONS.items()],
                        index=0 if not current[0] else (
                            ["（未选择）"] + list(REJECTION_REASONS.keys())
                        ).index(current[0]) if current[0] in REJECTION_REASONS else 0,
                        key=f"reason_{idx}_{name}",
                    )
                with col2:
                    note = st.text_input(
                        "备注（选填）",
                        value=current[1],
                        key=f"note_{idx}_{name}",
                    )
                if reason != "（未选择）":
                    for code, (label, _) in REJECTION_REASONS.items():
                        if label == reason:
                            st.session_state.rejection_reasons_map[name] = (code, note)
                            break

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ 已采纳", len(st.session_state.adopted))
    col2.metric("📋 备选", len(st.session_state.rejected))
    col3.metric(
        "⏳ 未处理",
        len(all_evals) - len(st.session_state.adopted) - len(st.session_state.rejected),
    )


def render_export():
    st.header("📦 导出清单")

    if not st.session_state.output:
        st.info("还没有搜索结果")
        return

    all_evals = []
    for cr in st.session_state.output.cuisines.values():
        all_evals.extend(cr.locked)

    # Feedback submission
    reason_map = st.session_state.rejection_reasons_map
    pending = [
        (name, code, note) for name, (code, note) in reason_map.items()
        if code and code in REJECTION_REASONS
    ]
    if pending:
        st.subheader("📝 待提交反馈")
        feedback_df = pd.DataFrame([
            {
                "菜品": name,
                "原因": REJECTION_REASONS[code][0],
                "备注": note,
            }
            for name, code, note in pending
        ])
        st.dataframe(feedback_df, hide_index=True)

        if st.button("📤 提交反馈", type="primary"):
            rejections = []
            for name, code, note in pending:
                eval_item = None
                for e in all_evals:
                    if e.proposal.name == name:
                        eval_item = e
                        break
                rejection = RejectionFeedback(
                    proposal_name=name,
                    proposal_name_cn=eval_item.proposal.name_cn if eval_item else "",
                    cuisine=eval_item.proposal.cuisine if eval_item else "",
                    reason_code=code,
                    reason_label=REJECTION_REASONS[code][0],
                    custom_note=note,
                )
                rejections.append(rejection)

            feedback_path = Path("data/feedback")
            saved = _save_feedback(rejections, feedback_path)
            st.session_state.rejection_reasons_map = {}
            st.toast(f"✅ 已提交 {len(rejections)} 条反馈到 {saved.name}")
            st.rerun()

        st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ 采纳清单")
        adopted_items = [
            e for e in all_evals if e.proposal.name in st.session_state.adopted
        ]
        if adopted_items:
            adopted_df = pd.DataFrame(
                [
                    {
                        "菜品": e.proposal.name,
                        "菜系": e.proposal.cuisine,
                        "建议售价": f"SGD {e.proposal.price_sgd:.2f}",
                        "总分": f"{e.total_score:.1f}",
                        "差异化理由": e.proposal.differentiation,
                    }
                    for e in adopted_items
                ]
            )
            st.dataframe(adopted_df, hide_index=True)

            csv = adopted_df.to_csv(index=False)
            st.download_button(
                "📥 导出采纳清单 CSV",
                csv,
                f"adopted_{pd.Timestamp.now():%Y-%m-%d}.csv",
                "text/csv",
            )
        else:
            st.caption("在「推荐结果」页勾选菜品后，这里会出现采纳清单")

    with col2:
        st.subheader("📋 备选清单")
        rejected_items = [
            e for e in all_evals if e.proposal.name in st.session_state.rejected
        ]
        if rejected_items:
            rejected_df = pd.DataFrame(
                [
                    {
                        "菜品": e.proposal.name,
                        "菜系": e.proposal.cuisine,
                        "总分": f"{e.total_score:.1f}",
                    }
                    for e in rejected_items
                ]
            )
            st.dataframe(rejected_df, hide_index=True)

            csv = rejected_df.to_csv(index=False)
            st.download_button(
                "📥 导出备选清单 CSV",
                csv,
                f"backup_{pd.Timestamp.now():%Y-%m-%d}.csv",
                "text/csv",
            )


def main():
    init_session()

    # Transfer completed output to session state for results/export tabs
    progress = _read_progress()
    if progress is not None and progress.get("done") and not progress.get("error") and _OUTPUT_FILE.exists():
        if st.session_state.output is None:
            st.session_state.output = _load_output()

    st.title("🔍 InstaChef SKU Explorer")
    st.caption("智能选品探索工具 · Generator → Evaluator 双 Agent 循环")

    render_control_panel()

    tab1, tab2, tab3 = st.tabs(
        [
            "🔍 搜索进度",
            "📋 推荐结果",
            "📦 导出清单",
        ]
    )

    with tab1:
        render_progress()
    with tab2:
        render_results()
    with tab3:
        render_export()

    # Auto-refresh while pipeline is running
    progress = _read_progress()
    if progress is not None and not progress.get("done", True):
        if progress.get("error"):
            _clear_progress()
            st.rerun()
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
