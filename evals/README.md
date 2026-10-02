# 评估集

一份"已经确定系统该怎么表现"的具体情形清单。改代码、改 prompt、换模型之前和之后各跑一次，就知道是变好还是变坏。

## 这里有什么

| 文件 | 评什么 | 怎么跑 |
|---|---|---|
| `cases/guard.yaml` | guard 对一道菜该拦还是该放 | 随 `pytest tests/` 每次都跑 |
| `cases/duplicate.yaml` | 去重检查该不该判重复 | 随 `pytest tests/` 每次都跑 |
| `cases/evaluator.yaml` | Evaluator 的结论和人工是否一致、多次运行是否稳定 | `python evals/run_evaluator.py`（调真实接口，花钱） |

前两个是确定性的，答案只有对错。第三个评的是模型，同一输入每次结果可能不同，所以要多跑几次看两件事：

- **一致率**：和人工标签一致吗。只统计有标签的菜。
- **稳定性**：同一道菜每次结论一样吗。不需要标签。

## 怎么加一条用例

1. 遇到一个错误结论（bug、人工拒绝、自己想到的边界情况），先把"正确的结论应该是什么"写成一条用例。
2. 跑一次，确认它失败。
3. 修代码，直到它通过。

暂时不打算修的问题，在用例上写 `known_gap` 说明原因。它会以"预期失败"的方式留在清单里；哪天修好了，测试会提醒你把这个标记删掉。

`evaluator.yaml` 里 `label` 为空的菜还没有人判断过。请业务专家填上 `accept` 或 `reject`（拒绝的写上 `reason`）。目前只有拒绝的标签，没有采纳的，所以一致率只能说明"人拒绝的菜，Evaluator 拦住了多少"。

## 跑 Evaluator 评估

```bash
python evals/run_evaluator.py                   # 每道菜评 3 次，按 Evaluator 当前配置
python evals/run_evaluator.py --thinking both   # 对比思考模式开和关
python evals/run_evaluator.py --runs 5 --preset siliconflow-deepseek-v4-flash
python evals/run_evaluator.py --thinking off --temperature 1.0   # 看 temperature 对波动的影响
python evals/run_evaluator.py --threshold 65    # 模拟通过线被 orchestrator 下调之后的轮次
```

原始结果写在 `evals/results/`（不进 git）。Evaluator 的联网搜索第一次真搜，之后回放 `evals/results/search_cache.json`，所以多次运行比较的是模型而不是搜索引擎；想刷新就删掉这个文件。

## 基线（2026-10-02）

`deepseek-v4-flash`，22 道菜（10 道有人工"拒绝"标签），每种配置各评 3 次。

| | 思考开 | 思考关 |
|---|---|---|
| 人工拒绝的 10 道里，3 次都没放行的 | 5 道 | 8 道 |
| 3 次之间结论翻转的菜 | 7 / 22 | 0 / 22 |
| 同一道菜的分数波动（中位数） | 7.5 分 | 0 分 |
| 每批耗时 | 32 秒 | 6 秒 |
| token（输入 + 输出） | 139k + 105k | 47k + 26k |
| 总分与自己的分项分对不上的结论 | 4 条 | 6 条 |

读这张表时注意：

- 标签只有拒绝的。一个什么都不放行的 Evaluator 在第一行能拿满分，所以这一行不能单独用来判断哪种配置更好。
- 两道因"口味不适合新加坡市场"被人工拒绝的墨西哥菜，两种配置下 3 次都放行了。
- 新马菜系那一批的总分被模型按 0–10 报出（3.25 而不是 32.5），两种配置都有。

同一天的两个补充实验（都是思考关）：

- temperature 从 0.1 调到 1.0：分数波动中位数从 0 升到 4.0 分。思考模式下 temperature 不生效，这是思考开时不稳定的主要来源之一。
- 通过线从 80 降到 65：分数不变，但人工拒绝的 10 道里放行的从 2 道变成 8 道。
