# InstaChef SKU Explorer

智能售货机选品探索工具 — 基于多 Agent 协作的菜品推荐系统。

Generator → Evaluator 双 Agent 循环，Orchestrator 协调，支持 Tavily 网络搜索、多层证据校验、HITL 人类反馈闭环。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                           │
│  控制面板 | 搜索进度 | 推荐结果 | 导出清单 | HITL 反馈         │
├─────────────────────────────────────────────────────────────────┤
│                      Orchestrator                               │
│          (状态机 + LLM 语义压缩 + 按 SKU 稀缺度分配配额)          │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│ 2-Hop    │ Guard    │ Evaluator│ Generator│ HITL               │
│ Search   │ 高压线   │ 评分审计  │ 提案生成  │ 反馈引擎           │
│ Tavily   │ 确定性   │ 4 层约束  │ LLM      │ 黑名单 + RAG       │
├──────────┴──────────┴──────────┴──────────┴────────────────────┤
│              Audit Trail (data/logs/)                            │
│  evaluation | tracing | lineage | executive_summary | feedback  │
└─────────────────────────────────────────────────────────────────┘
```

## 特性

- **多 Agent 协作**: Generator 生成提案 → Evaluator 评估打分 → Orchestrator 协调
- **四层过滤**: 数据谱系验证 → HITL 黑名单 → Guard 高压线 → Evaluator 审计
- **2-Hop 搜索**: Tavily 宽搜发现 → 提取餐厅名 → 定向补证据
- **双轨证据**: dish_name_matched（严格）+ trend_matched（宽松）
- **HITL 反馈**: 人工拒绝 → 黑名单 auto-veto → Generator prompt 学习
- **配额与选菜分离**: 每批总数按各菜系现有 SKU 数分配（SKU 越少配额越多）；菜系内按菜品分取前 N 名，不设及格线
- **智能 Orchestrator**: 配额未满时由 LLM 诊断失败原因并回灌给 Generator
- **性能追踪**: 5 阶段计时 tracing，瓶颈可视化
- **中英双语**: 所有提案同时包含中英文名称和描述

## 快速开始

### 环境要求

- Python 3.11+
- [Tavily API Key](https://tavily.com/)（搜索）
- [SiliconFlow API Key](https://siliconflow.cn/)（LLM，默认 DeepSeek-V3）

> 🎁 **免费白嫖国产大模型**：通过 [这个邀请链接](https://cloud.siliconflow.cn/i/SV01waIi) 注册硅基流动，完成支付宝实名认证即可领取代金券，零成本体验 MiniMax-2.5、GLM-5.1、DeepSeek V3 等国产顶流模型。注册后 API Key 即开即用。

### 1. 克隆仓库

```bash
git clone git@github.com:XploreXing/InstaChef_SKU_Explorer.git
cd InstaChef_SKU_Explorer
```

### 2. 创建环境

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 3. 配置 API Keys

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Keys：

```env
LLM_API_KEY=sk-your-siliconflow-key
TAVILY_API_KEY=tvly-your-tavily-key
```

`.env` 文件已加入 `.gitignore`，不会被提交。

### 4. 启动

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

### 5. 运行测试

```bash
pytest tests/ -v
# 125 passed
```

## 配置

编辑 `config.yaml` 调整参数。Web UI 侧边栏"模型配置"可动态覆盖模型选择。

```yaml
orchestrator:
  max_rounds_per_cuisine: 3
  total_target: 10          # 一批总共找几道，按各菜系现有 SKU 数分配
  cuisines: ["Chinese", "Japanese", "Korean", "Thai",
             "Singaporean/Malay", "Mexican"]

llm:
  base_url: "https://api.siliconflow.cn/v1"
  enrichment_model: "deepseek-ai/DeepSeek-V3"
  generator_model: "deepseek-ai/DeepSeek-V3"
  evaluator_model: "deepseek-ai/DeepSeek-V3"
```

## 项目结构

```
SKU_Explorer/
├── app.py                    # Streamlit 入口
├── orchestrator.py           # 总控：状态机 + Agent 协调
├── models.py                 # 所有 dataclass 定义
├── config.yaml               # 默认配置
├── CLAUDE.md                 # 开发规范
├── agents/
│   ├── generator.py          # Generator Agent (提案生成)
│   └── evaluator.py          # Evaluator Agent (评分审计)
├── prompts/
│   ├── generator_system.md   # Generator 系统提示词
│   └── evaluator_system.md   # Evaluator 系统提示词
├── utils/
│   ├── search.py             # Tavily 搜索 + 质量过滤
│   ├── guard.py              # 确定性高压线
│   ├── data_loader.py        # CSV 数据加载 + LLM 标签
│   └── feedback_loader.py    # HITL 反馈加载
├── data/
│   ├── input/                # 源数据 (CSV)
│   ├── cache/                # LLM 标签缓存
│   ├── runtime/              # 临时文件 (gitignored)
│   └── logs/                 # 审计日志 (gitignored)
└── tests/                    # 125 个单元测试
```

## 核心流程

### 单轮 Pipeline

```
Search (2-hop Tavily)
  → Guard (确定性规则拦截: 清真/冷食/热柜限制)
    → HITL Blacklist (人类反馈去重)
      → Evaluator (搜索验证真实性 → 趋势热度、小贩替代性 2 维打分)
        → Judging (代码计算菜品分，菜系内取前 N 名)
          → Orchestrator (配额未满时：LLM 诊断 → feedback → 下一轮)
```

### 审计日志

| 文件 | 内容 |
|------|------|
| `evaluation_<ts>.json` | 每个提案的打分/否决详情 |
| `tracing_<ts>.json` | 各阶段耗时分解 (search/generate/evaluate/feedback) |
| `lineage_<ts>.json` | 搜索引用→提案证据链路 (source_refs → URL → evidence_hits) |
| `executive_summary_<ts>.json` | LLM 生成的高管摘要 |
| `search_snippets_<ts>.json` | Tavily 原始搜索结果 |

### HITL 反馈循环

```
用户拒绝菜品 → Tab 3 提交
  → data/feedback/rejections_<ts>.json
    → 黑名单: 精确菜名去重 (fast-fail)
    → Generator prompt: "避免做这些菜"
```

## 开发

```bash
# 测试（TDD 模式）
pytest tests/ -v

# 提交（遵循 Conventional Commits）
# 格式: <type>(<scope>): <description>
# 类型: feat, fix, refactor, test, docs, chore
```

## License

MIT
