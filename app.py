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
_PROGRESS_FILE = Path(__file__).resolve().parent / "data" / "pipeline_progress.json"
_OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "pipeline_output.json"


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
                source_refs=p_data.get("source_refs", []),
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


def init_session():
    defaults = {
        "output": None,
        "adopted": set(),
        "rejected": set(),
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
            st.text_input(
                "API Base URL",
                value="https://api.siliconflow.cn/v1",
                key="api_base_url",
                help="模型供应商的 API 地址",
            )
            st.text_input(
                "API Key",
                type="password",
                value=os.getenv("LLM_API_KEY", ""),
                key="api_key",
                help="留空则使用 .env 中的 LLM_API_KEY",
            )
            st.text_input(
                "Enrichment Model",
                value="deepseek-ai/DeepSeek-V3",
                key="enrichment_model_name",
                help="用于给商品打菜系标签",
            )
            st.text_input(
                "Generator Model",
                value="deepseek-ai/DeepSeek-V3",
                key="generator_model_name",
                help="用于生成菜品提案",
            )
            st.text_input(
                "Evaluator Model",
                value="deepseek-ai/DeepSeek-V3",
                key="evaluator_model_name",
                help="用于评估菜品打分",
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
            st.rerun()

        if start_clicked:
            st.session_state.output = None
            st.session_state.adopted = set()
            st.session_state.rejected = set()
            _clear_progress()

            # Capture sidebar params before starting thread
            # (st.session_state is not thread-safe)
            target = st.session_state.get("target", 10)
            threshold = st.session_state.get("threshold", 80)
            max_rounds = st.session_state.get("max_rounds", 3)
            api_base_url = st.session_state.get("api_base_url", "")
            api_key = st.session_state.get("api_key", "")
            enrichment_model = st.session_state.get("enrichment_model_name", "")
            generator_model = st.session_state.get("generator_model_name", "")
            evaluator_model = st.session_state.get("evaluator_model_name", "")

            state = {
                "done": False,
                "messages": [],
                "error": None,
                "locked_count": 0,
                "target": target,
                "phase": "",
                "cuisine": "",
                "stop_requested": False,
            }
            _write_progress(state)

            def run_pipeline():
                def log(msg):
                    state["messages"].append(msg)
                    _write_progress(state)
                    print(msg, flush=True)

                def update_state(cuisine, round_num, phase, meta):
                    state["phase"] = phase
                    state["cuisine"] = cuisine
                    state["locked_count"] = meta.get("locked_count", 0)
                    state["target"] = meta.get("remaining", 0) + meta.get("locked_count", 0)
                    _write_progress(state)

                try:
                    orch = Orchestrator("config.yaml")
                    # Override config with sidebar parameters
                    orch.config["orchestrator"]["target_per_cuisine"] = target
                    orch.config["orchestrator"]["pass_threshold"] = threshold
                    orch.config["orchestrator"]["max_rounds_per_cuisine"] = max_rounds

                    # Override LLM config with user-provided values (empty = keep default)
                    if api_base_url:
                        orch.config["llm"]["base_url"] = api_base_url
                    if api_key:
                        orch.config["llm"]["api_key"] = api_key
                    if enrichment_model:
                        orch.config["llm"]["enrichment_model"] = enrichment_model
                    if generator_model:
                        orch.config["llm"]["generator_model"] = generator_model
                    if evaluator_model:
                        orch.config["llm"]["evaluator_model"] = evaluator_model

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
                        log("⏹️ 用户中断。已保存当前进度。")
                    else:
                        _save_output(output)
                        log_path = _save_evaluation_log(output, Path("data"))
                        log(f"📝 评估日志已保存: {log_path.name}")
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
    locked = progress.get("locked_count", 0)
    target_val = progress.get("target", 10)
    phase = progress.get("phase", "")
    cuisine = progress.get("cuisine", "")

    phase_label = {
        "searching": f"🔍 正在搜索 {cuisine} 菜系趋势...",
        "summarizing": f"📝 正在整理 {cuisine} 搜索结果...",
        "generating": f"🤖 正在生成 {cuisine} 菜品提案...",
        "evaluating": f"📊 正在评估 {cuisine} 菜品提案...",
    }.get(phase, f"⏳ {phase}...")

    st.info(f"{phase_label}")
    if target_val > 0:
        st.progress(min(locked / target_val, 1.0), text=f"已锁定 {min(locked, target_val)}/{target_val}")
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
        # Use English name as the stable key (not the display name which combines CN+EN)
        real_name = all_evals[idx].proposal.name
        if row["采纳"]:
            st.session_state.adopted.add(real_name)
            st.session_state.rejected.discard(real_name)
        else:
            st.session_state.rejected.add(real_name)
            st.session_state.adopted.discard(real_name)

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
