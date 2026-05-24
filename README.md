# InstaChef SKU Explorer

智能选品探索工具 — Generator → Evaluator 双 Agent 循环，从外部餐饮趋势中自动发现并评估潜在新 SKU。

## 快速启动

```bash
pip install -r requirements.txt
export LLM_API_KEY="your-api-key"
export TAVILY_API_KEY="your-tavily-key"
streamlit run app.py
```

## 架构

Orchestrator (Python 状态机) → Generator (搜索+提案) → Evaluator (约束检查+打分) → 循环最多 3 轮/菜系 → Top 10 输出

## 配置

所有配置在 `config.yaml`，包括 LLM 端点、搜索来源、菜系列表、打分阈值。
