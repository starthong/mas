# mas

多智能体协作系统 + 工具调用 JSON 格式的 LoRA 微调实验。

系统分 Planner、Executor、Reviewer 三个角色，用 LangGraph 串成
`plan -> execute -> review -> (replan)` 流程。训练用 LLaMA-Factory 做 QLoRA，
评测统计 JSON 可解析率、schema 合规率、工具名准确率和参数完全匹配率。

## 目录

```
mas/
├── configs/                 # Kaggle QLoRA 配置
├── data/                    # 训练/测试数据、系统任务
├── notebooks/               # Kaggle notebook
├── report/                  # Overleaf 用的 ctexart 模板
├── scripts/                 # 环境、数据、评测脚本
├── src/
│   ├── agents/              # planner / executor / reviewer
│   ├── experiments/         # single / sequential 模式 runner
│   ├── graph/               # LangGraph 工作流
│   ├── llm/                 # OpenAI 与本地 HF 两种后端
│   ├── tools/               # 工具注册表
│   └── eval/                # 指标与评测入口
└── tests/
```

## Linux 开发环境（Ubuntu）

Ubuntu：

```bash
sudo apt update
sudo apt install -y git curl build-essential openssh-server
cd /data
git clone git@github.com:starthong/mas.git
cd mas
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

跑测试：

```bash
pytest -q
```

开发和调试都在 Ubuntu 里做，Kaggle 只当远程 GPU 节点：代码先在 Ubuntu 改好、
测试、push 到 GitHub；Kaggle clone 后只负责训练；adapter 和结果下载回 Ubuntu
做分析、画图和写报告。这样本地 Linux 环境和 Kaggle GPU 是两条线，不会互相替代。

环境信息可以留下来放进报告附录：

```bash
uname -a
lsb_release -a
python -V
pip list | head -20
```

## Kaggle 训练

默认用 Qwen2.5-3B-Instruct。T4 只有 16GB，3B 的 QLoRA 和后续评测都更稳；
7B 也能跑，但要改 `configs/train_qlora_kaggle.yaml` 里的模型路径，并把
`cutoff_len` 降到 1024 左右。

1. 新建 Notebook，Settings 里选 **GPU T4 x2**，打开 **Internet**。
2. Add-ons -> Secrets，添加 `GITHUB_TOKEN`，值是能读取本仓库的 GitHub PAT。
3. 把 `notebooks/kaggle_qlora.ipynb` 导入 Kaggle，或者直接上传这个文件。
4. Run All。

notebook 会做这几件事：

- clone 仓库到 `/kaggle/working/mas`
- 下载模型到 `/kaggle/temp/models/Qwen2.5-3B-Instruct`
- 生成 `data/toolcall_sft.jsonl` 和 `data/test/toolcall_test.jsonl`
- 用 `configs/train_qlora_kaggle.yaml` 训练
- 用本地 HF 后端分别评测 base 和 LoRA

训练输出在 `/kaggle/working/output/qwen25-3b-toolcall-lora`，评测结果在
`/kaggle/working/mas/results/`。会话结束前把 adapter 和两个 JSON 下载下来。

Kaggle 免费 GPU 大约每周 30 小时，单次会话有时长上限，训练和评测不要卡着
上限跑。第一次运行建议把 `--limit` 设成 20 先看流程通不通。

## 数据

```bash
python scripts/build_toolcall_dataset.py
```

默认生成约 700 条训练样本和 40 条测试样本，覆盖天气、计算器、搜索和
无需工具四类。模板在脚本里，改起来比较直接。正式实验前建议扩到 1000 条以上，
并按工具划分训练/测试，避免模板泄漏。

`data/samples/toolcall_sample.jsonl` 只是三条格式样例，不要拿它直接训练。

## 评测

用 `--backend` 指定后端：

```bash
# OpenAI 兼容接口
python -m src.eval.run_eval --backend openai --model toolcall \
  --data data/test/toolcall_test.jsonl --out results/ft.json

# 本地 transformers + peft
python -m src.eval.run_eval --backend hf \
  --base-model /kaggle/temp/models/Qwen2.5-3B-Instruct \
  --adapter /kaggle/working/output/qwen25-3b-toolcall-lora \
  --data data/test/toolcall_test.jsonl --out results/ft.json --limit 100
```

base 模型去掉 `--adapter` 再跑一遍，得到 `results/base.json`。

## 多智能体实验

`data/system_tasks.jsonl` 里有 31 条任务：24 条单步任务 + 7 条多步任务。runner 支持 `single` 和 `sequential`，`--planner` 可选 `lora/base`，`--reviewer` 可选 `lora/base/none`。`ft` 仍可作为 `lora` 的别名。

```bash
python -m src.experiments.run_modes \
  --backend hf \
  --base-model /kaggle/temp/models/Qwen2.5-3B-Instruct \
  --adapter /kaggle/input/datasets/starthong/mas-3b-toolcall-output \
  --tasks data/system_tasks.jsonl \
  --modes single,sequential \
  --reviewer lora \
  --runs 3 \
  --temperatures 0 \
  --out results/modes-lora-0.json
```

Reviewer / Planner 消融：

```bash
# 无 Reviewer
--reviewer none

# base Reviewer + LoRA Planner/Executor
--reviewer base

# base Planner + LoRA Executor
--planner base

# 角色分离：base Planner + LoRA Executor + base Reviewer
--planner base --reviewer base

# 全 LoRA（默认）
--planner lora --reviewer lora
```

第一次先跑 2 条确认链路：

```bash
python -m src.experiments.run_modes --backend hf \
  --base-model /kaggle/temp/models/Qwen2.5-3B-Instruct \
  --adapter /kaggle/input/datasets/starthong/mas-3b-toolcall-output \
  --tasks data/system_tasks.jsonl --modes single,sequential \
  --reviewer lora --runs 1 --temperatures 0 --limit 2 \
  --out results/modes-smoke.json
```

base 模型去掉 `--adapter`，输出到 `results/modes-base.json`，然后对比两边的成功率、JSON 合法率、工具匹配和延迟。

## 报告

`report/main.tex` 用 `ctexart`，在 Overleaf 里编译：

1. New Project -> Blank Project
2. 上传 `main.tex`、`refs.bib`、`figures/`
3. Menu -> Compiler -> **XeLaTeX**

## 备注

- `search` 现在查本地 20 篇小文档，关键词/字符匹配，不依赖外部 API。
- Executor 一个子任务只调用一次工具；多步任务靠 Planner 拆成多个子任务，还不是完整 ReAct 循环。
- 训练数据是模板合成的，适合验证格式，不足以证明真实任务能力。
- 报告里需要写清楚：协作模式对比、LoRA 前后对比、稳定性指标，三者缺一不可。
