import sys
import time
import threading
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator

st.set_page_config(
    page_title="InstaChef SKU Explorer",
    page_icon="🔍",
    layout="wide",
)


def init_session():
    defaults = {
        "output": None,
        "adopted": set(),
        "rejected": set(),
        "pipeline_thread": None,
        "pipeline_holder": None,  # {"done": bool, "messages": list, "output": Optional, "error": Optional}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_control_panel():
    with st.sidebar:
        st.header("⚙️ 控制区")

        cuisines = st.multiselect(
            "选择菜系",
            options=["中式", "日式", "韩式", "泰式", "新马", "墨西哥"],
            default=["墨西哥"],
        )

        with st.expander("⚡ 高级参数"):
            st.slider("目标数/菜系", 1, 20, 10, key="target")
            st.slider("及格线", 50, 100, 80, key="threshold")
            st.slider("最大轮数", 1, 5, 3, key="max_rounds")

        running = (
            st.session_state.pipeline_holder is not None
            and not st.session_state.pipeline_holder["done"]
        )
        if st.button(
            "🚀 启动搜索" if not running else "⏳ 搜索中...",
            type="primary",
            use_container_width=True,
            disabled=running,
        ):
            st.session_state.adopted = set()
            st.session_state.rejected = set()
            st.session_state.pipeline_holder = {
                "done": False,
                "messages": [],
                "output": None,
                "error": None,
            }

            def run_pipeline():
                try:
                    orch = Orchestrator("config.yaml")
                    orch.load_skus()

                    orch.state_callbacks["on_state_change"].append(
                        lambda c, r, p, m: st.session_state.pipeline_holder[
                            "messages"
                        ].append(
                            f"🔍 **{c}** · Round {r} · `{p}` · "
                            f"已锁定 {m.get('locked_count', 0)}/"
                            f"{m.get('remaining', 10) + m.get('locked_count', 0)}"
                        )
                    )
                    orch.state_callbacks["on_round_complete"].append(
                        lambda res: st.session_state.pipeline_holder[
                            "messages"
                        ].append(
                            f"✅ **{res.cuisine}** Round {res.round_num}: "
                            f"生成 {res.proposals_generated} → "
                            f"✅{res.passed_count} / ❌{res.rejected_count} "
                            f"({res.elapsed_seconds:.1f}s)"
                        )
                    )

                    st.session_state.pipeline_holder["output"] = orch.run(cuisines)
                except Exception as e:
                    st.session_state.pipeline_holder["error"] = str(e)
                finally:
                    st.session_state.pipeline_holder["done"] = True

            thread = threading.Thread(target=run_pipeline, daemon=True)
            st.session_state.pipeline_thread = thread
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
    holder = st.session_state.pipeline_holder

    if holder is None:
        st.info("点击侧边栏 🚀 启动搜索 开始探索")
        return

    if holder["error"]:
        st.error(f"运行出错: {holder['error']}")
        return

    if holder["done"] and holder["output"]:
        st.success("✅ 搜索完成！切换到「📋 推荐结果」查看")
        for msg in holder["messages"]:
            st.write(msg)
        st.caption(
            f"总耗时: {holder['output'].total_elapsed_seconds:.0f}s"
        )
        return

    # Still running — show current progress
    st.info("⏳ 搜索运行中，进度每 2 秒自动刷新...")
    for msg in holder["messages"]:
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
        rows.append({
            "采纳": name in st.session_state.adopted,
            "排名": i + 1,
            "菜品": name,
            "菜系": e.proposal.cuisine,
            "蓝海(40)": f"{e.cuisine_blue_ocean * 4:.0f}",
            "趋势(35)": f"{e.trend_heat * 3.5:.0f}",
            "替代(25)": f"{e.hawker_substitutability * 2.5:.0f}",
            "总分": f"{e.total_score:.1f}",
            "描述": e.proposal.description[:80],
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

    for _, row in edited.iterrows():
        name = row["菜品"]
        if row["采纳"]:
            st.session_state.adopted.add(name)
            st.session_state.rejected.discard(name)
        else:
            st.session_state.rejected.add(name)
            st.session_state.adopted.discard(name)

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

    # --- auto-refresh: if pipeline is running, poll every 2s ---
    holder = st.session_state.pipeline_holder
    if holder is not None and not holder["done"]:
        if holder["error"]:
            st.session_state.pipeline_holder = None
            st.rerun()
        time.sleep(2)
        st.rerun()

    # --- pipeline just finished — transfer output to session state ---
    if holder is not None and holder["done"] and holder["output"]:
        st.session_state.output = holder["output"]
        st.session_state.pipeline_holder = None

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


if __name__ == "__main__":
    main()
