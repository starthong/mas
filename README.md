# mas

多智能体工具调用研究原型：Planner / Executor / Reviewer 三角色协作，
以及面向工具调用 JSON 协议的 QLoRA 微调实验。

项目面向有限、可复现的工具集，重点不是构建通用 Agent 平台，而是研究三个
问题：LoRA 能否稳定多智能体链路中的工具调用；不同协作拓扑是否真的提升
任务完成率、代价是什么；温度、数据规模和随机种子是否改变结论。

## 命名约定

- `v1`：750 条主训练集这一数据配方，不是模型版本；
- `v2`：1,124 条扩展训练集这一数据配方，不是 v1 的 checkpoint；
- `expanded`：55 条系统任务这一任务集，和 v1/v2 adapter 无关；
- 报告中的 v1/v2 比较是两组独立训练的数据配方对照，不是同一组参数的两个快照；
- v1 是预先设定的基线，v2 是后续数据扩展消融；系统任务沿用 v1，不在系统结果出来后重新选 adapter。

## 系统

- `Planner`：把任务拆成 1–5 个有序子任务；
- `Executor`：每个子任务选择并调用一个工具，输出
  `{"name": ..., "arguments": ...}`；
- `Reviewer`：检查结果是否完整，返回 `pass`、`score`、`issues` 和
  `suggestion`，不通过时触发有限轮重规划。

工作流由 LangGraph 状态图实现，支持四种拓扑：

| 模式 | 结构 |
|---|---|
| `single` | 单 Executor，直接调用工具 |
| `sequential` | Planner → Executor → Reviewer，可重规划 |
| `supervisor` | Supervisor 在子任务之间决定继续执行、审查或结束 |
| `debate` | 两个 Planner 提出计划，Judge 选择后执行和审查 |

工具层包含天气、计算器、本地检索和 `final_answer` 四类；天气和计算器是
确定性 mock，检索基于 20 篇本地文档的关键词/字符重叠。工具通过 JSON Schema
注册，调用结果和错误写入共享状态。

## 数据与模型

| 数据 | 规模 | 用途 |
|---|---:|---|
| 主训练集 | 750 | 天气、计算、检索、无需工具回答 |
| 扩展训练集 | 1,124 | 增加城市、主题和随机算式，用于数据配方对照 |
| 简单格式测试 | 41 | 模板内样本，base 已接近饱和 |
| 难例测试 | 57 | 未见城市/主题、缺参、格式诱导 |
| 未见工具 OOD | 40 | 4 个训练中未出现的工具 |
| 系统任务 | 31 | 24 条单步 + 7 条多步 |
| 扩展系统任务 | 55 | 41 条单步 + 14 条多步，用于样本量复核 |

基座模型为 Qwen2.5-3B-Instruct，训练使用 4-bit QLoRA（rank 16，alpha 32，
2 epoch，T4）。主 adapter 位于 Kaggle Dataset
`starthong/mas-3b-toolcall-output`；报告中的 v1/v2 对照各训练 3 个 adapter，
`configs/seeds/` 已扩展到 seed 42–46，seed 45/46 用于 5-seed 复核。
训练和测试按场景划分，避免同一模板同时出现在两边。

## 评测

格式评测区分 JSON 可解析率、schema 合规率、工具名准确率和参数完全匹配率；
系统评测区分工具调用正确性与任务完成。主要对照包括：

- base zero-shot、base + few-shot、LoRA；
- single / sequential / supervisor / debate；
- LoRA / base Planner、LoRA / base / 无 Reviewer；
- 温度 0 / 0.7；
- 750 与 1,124 数据配方，各 5 个 seed（仅格式评测）。

补充实验入口：

| 脚本 | 内容 |
|---|---|
| `scripts/run_system_expanded.sh` | 在 55 条扩展系统任务上跑 sequential/supervisor/debate（使用主 adapter） |
| `scripts/run_sensitivity.sh` | 在 14 条多步任务上比较 max_rounds=1/2/3 |
| `scripts/run_constrained.sh` | vLLM guided-JSON 约束解码基线（base/LoRA，hard/OOD） |
| `scripts/summarize_modes.py` | 从 run_modes 输出汇总成功率、Wilson 区间和延迟 |
| `scripts/analyze_seeds.py` | 汇总 5 个 seed 的均值、标准差和差值 |
| `configs/seeds/*-seed45.yaml`、`*-seed46.yaml` | v1/v2 的 5-seed 复核配置 |

这些脚本用于扩大样本量和参数敏感性检验，结果已写入报告和 `results/review/`。

31 条系统任务只有 7 条多步任务，55 条扩展集把多步任务增加到 14 条。
报告给出 Wilson 95% 置信区间，并将系统成功率作为描述性结果，而不是
“显著优于”的证据。5 个 seed 只用于 v1/v2 格式对照；系统任务只跑了
1 轮，温度稳定性另行跑了 3 次。

## 主要结果

| 指标 | 结果 |
|---|---|
| hard schema，base → LoRA | 94.7% → 98.2% |
| hard 工具名，base → LoRA | 86.0% → 91.2% |
| base + few-shot | schema 96.5%，工具名 82.5% |
| OOD 工具名，base → LoRA | 82.5% → 90.0% |
| base single → sequential | 67.7% → 51.6% |
| LoRA sequential / few-shot | 80.6% / 83.9% |
| supervisor | 成功率相同，延迟约 +64% |
| debate | 80.6%，延迟 40.94 s |
| 温度 0.7 | JSON/schema 100%，任务成功率 74.2% |
| 5-seed hard schema | v1 .979±.008；v2 .975±.020 |
| 5-seed hard 工具名 | v1 .895±.012；v2 .902±.016 |
| 55 条 sequential/supervisor | 74.5%；debate 72.7% |
| max_rounds 1/2/3 | 14 条多步均为 42.9%；延迟 20.43→29.27 s |

这些结果说明 LoRA 改善的是工具调用协议和工具选择，并不能替代规划与审查；
增加角色或数据也不会自动提高任务完成率。

## 仓库结构

```
mas/
├── configs/        QLoRA、扩展数据和 seed 配置
├── data/           训练集、三套格式测试集、系统任务和样例
├── notebooks/      Kaggle 训练/评测 notebook
├── report/         ctexart 技术报告、图和 refs.bib
├── results/        逐样本 JSON 记录和 summary
├── scripts/        数据生成、评测、消融和打包脚本
├── src/
│   ├── agents/     planner / executor / reviewer / supervisor / debate
│   ├── eval/       指标与格式评测入口
│   ├── experiments/ single、四拓扑和消融 runner
│   ├── graph/      LangGraph 工作流
│   ├── llm/        OpenAI 与本地 transformers/PEFT 后端
│   └── tools/      工具注册表、内置工具、OOD 工具
└── tests/          图流程和指标测试
```

## 复现材料

- 源码、训练/测试数据和原始评测记录都在仓库中；
- `results/` 保存每个配置的逐样本 JSON 和 summary，不只保留聚合表；
- `configs/` 保存主训练、扩展数据和 6 个 seed 配置；
- 主 adapter（约 114 MB）不放入 Git，保存在 Kaggle Dataset
  `starthong/mas-3b-toolcall-output`；
- 报告源文件为 `report/main.tex`，使用 XeLaTeX 编译。

主要入口：

| 入口 | 作用 |
|---|---|
| `python -m src.main` | 运行单条任务 |
| `python -m src.eval.run_eval` | 格式评测（支持 `--registry ood`） |
| `python -m src.experiments.run_modes` | 单 Agent/多拓扑实验 |
| `python scripts/analyze_results.py` | 置信区间、归一化 exact match、McNemar |

## 范围与限制

- 工具是 mock 或本地小语料，不能代表开放域真实工具；
- Executor 一个子任务只调用一次工具，不是完整 ReAct 循环；
- 微调只覆盖单步工具调用协议，没有 Planner/Reviewer 专用微调数据；
- 没有 JSON mode / grammar-constrained decoding 基线；
- 系统任务 31 条、多步 7 条，扩展后 55 条、多步 14 条，但仍各只跑 1 轮；
- 5-seed 对照只覆盖 v1/v2 格式评测，不覆盖全部系统任务。

项目是研究原型，不是生产级 Agent 平台。
