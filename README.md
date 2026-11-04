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

工具层包含天气、计算器、检索和 `final_answer` 四类。天气通过 Open-Meteo
实时查询；检索按 Serper、Tavily、Sogou、Wikipedia 的顺序调用外部服务，
可用 `SERPER_API_KEY` 或 `TAVILY_API_KEY` 提高稳定性；计算器和
`final_answer` 是本地确定性实现。工具通过 JSON Schema 注册，调用结果和
错误写入共享状态。OOD 工具只做 schema 注册，不执行真实操作。

## 数据与模型

| 数据 | 规模 | 用途 |
|---|---:|---|
| 主训练集 | 750 | 天气、计算、检索、无需工具回答 |
| 扩展训练集 | 1,124 | 增加城市、主题和随机算式，用于数据配方对照 |
| 简单格式测试 | 41 | 模板内样本，base 已接近饱和 |
| 难例测试 | 57 | 未见城市/主题、缺参、格式诱导 |
| 未见工具 OOD | 40 | 32 条四个未见工具 + 8 条 final 路由 |
| 核心系统子集 | 31 | 55 条开发集的子集（24 单步 + 7 多步） |
| 系统开发集 | 55 | 31 核心 + 24 扩展；41 单步 + 14 多步 |
| 系统 holdout | 25 | 与开发集不重叠；15 单步 + 10 多步 |
| Planner SFT | 1,392 | 合成规划数据，避开 55/holdout |
| Reviewer SFT | 2,784 | 上述任务的正/负审查样本 |

系统层共 **80 条互不重复任务**（55 开发 + 25 holdout），31 条不是独立的第三套测试集。

基座模型为 Qwen2.5-3B-Instruct，训练使用 4-bit QLoRA（rank 16，alpha 32，
2 epoch，T4）。主 adapter 位于 Kaggle Dataset
`starthong/mas-3b-toolcall-output`；报告中的 v1/v2 对照各训练 5 个 adapter，
`configs/seeds/` 已扩展到 seed 42–46。
训练和测试按场景划分，避免同一模板同时出现在两边。

## 评测

格式评测区分 JSON 可解析率、schema 合规率、工具名准确率和参数匹配率；
系统评测报告 **required-call completion**：预期工具/参数调用出现、工具无报错、
结果非空。它不检查多步调用顺序，也不对 `final_answer` 自由文本做语义评分，
因此 98.2% 等数字不是端到端答对率。主要对照包括：

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
| `scripts/build_planner_dataset.py` | 生成 Planner 专用 SFT 数据（与 55 条评测任务不重合） |
| `scripts/train_planner.sh` | 训练独立 Planner LoRA adapter |
| `scripts/build_reviewer_dataset.py` | 生成 Reviewer 正/负样本 |
| `scripts/train_reviewer.sh` | 训练独立 Reviewer LoRA adapter |
| `scripts/summarize_modes.py` | 从 run_modes 输出汇总成功率、Wilson 区间和延迟 |
| `scripts/analyze_seeds.py` | 汇总 5 个 seed 的均值、标准差和差值 |
| `configs/seeds/*-seed45.yaml`、`*-seed46.yaml` | v1/v2 的 5-seed 复核配置 |

这些脚本用于扩大样本量和参数敏感性检验，结果已写入报告和 `results/review/`。

31 条核心子集只有 7 条多步任务，包含在 55 条开发集中；55 条把多步任务增加到
14 条。报告给出 Wilson 95% 置信区间，并将 required-call completion 作为描述性
结果，而不是“显著优于”的证据。5 个 seed 只用于 v1/v2 格式对照；系统任务只跑了
1 轮，温度稳定性另行跑了 3 次。

## 主要结果

| 指标 | 结果 |
|---|---|
| hard schema，base → LoRA | 94.7% → 98.2% |
| hard 42 条工具必需工具名 | 40/42 → 41/42 |
| hard 15 条 final 路由 | 9/15 → 11/15 |
| OOD 32 条新工具工具名 | 32/32 → 32/32 |
| OOD 8 条 final 路由 | 1/8 → 4/8 |
| 31 条 base sequential / LoRA | 71.0% / 77.4% |
| 55 条共享 adapter，required-call | sequential/supervisor 98.2%，debate 96.4% |
| 55 条三角色 adapter | 94.5/92.7/94.5% |
| 25 条 holdout（三角色） | 25/25，三种模式 |
| max_rounds 1/2/3 | 14 条多步均为 85.7%（12/14）；延迟 24.64→27.11 s |
| 温度 0 / 0.7 | required-call completion 93.6% / 96.8% |
| 5-seed hard schema | v1 .979±.008；v2 .975±.020 |
| vLLM fp16 guided JSON | schema 100%；hard 工具样本 42/42；OOD 参数无提升 |

这些结果说明 LoRA 最明确的收益是结构协议遵循和部分 tool-vs-final 路由；
当前 32 条未见工具与结构化参数已经接近饱和，不能声称 LoRA 提升了未见工具选择。
98.2% 是 required-call completion，不是最终答案语义正确率。

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
- `configs/` 保存主训练、扩展数据、10 个 seed 配置（v1/v2 × seed 42–46）和 Planner/Reviewer 角色配置；
- 主 adapter（约 114 MB）不放入 Git，保存在 Kaggle Dataset
  `starthong/mas-3b-toolcall-output`；
- Planner adapter 保存在 `mas-planner-lora`，Reviewer adapter 保存在
  `mas-reviewer-lora`；两者同样不放入 Git；
- 报告源文件为 `report/main.tex`，使用 XeLaTeX 编译；图表使用 `[H]` 随文放置。

主要入口：

| 入口 | 作用 |
|---|---|
| `python -m src.main` | 运行单条任务 |
| `python -m src.eval.run_eval` | 格式评测（支持 `--registry ood`） |
| `python -m src.experiments.run_modes` | 单 Agent/多拓扑实验 |
| `python scripts/analyze_results.py` | 置信区间、归一化 exact match、McNemar |

## 范围与限制

- 天气和检索依赖外部服务；离线时需要 API key 或缓存，OOD 工具仍是 schema 级 mock；
- Executor 一个子任务只调用一次工具，不是完整 ReAct 循环；
- 微调只覆盖单步工具调用协议；Planner/Reviewer 使用单独的合成 SFT 数据（Planner 1,392 条，Reviewer 2,784 条），三角色 adapter 已完成 55 条和 25 条 holdout 对比；
- 约束解码只覆盖顶层 JSON schema，不约束完整工具参数；schema 可达 100%，但 OOD 结构化参数没有提升；
- 系统层是 55 条开发集（含 31 条核心子集）+ 25 条不重叠 holdout，共 80 条互不重复任务；31 不是独立第三套测试集；
- 系统指标是 required-call completion，不检查多步调用顺序，也不对 final_answer 自由文本做语义评分；“现在几点”返回“当前时间”仍会被记为完成；
- 5-seed 对照只覆盖 v1/v2 格式评测，不覆盖全部系统任务；
- 早期 Planner 干预没有提升 14 条多步任务；后续定位为 Executor 最终答案路由，修复后的 55/25 条结果见报告。

项目是研究原型，不是生产级 Agent 平台。
