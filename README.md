# mas

多智能体协作系统 + 工具调用 JSON 格式的 LoRA 微调实验。

系统分 Planner、Executor、Reviewer 三个角色，用 LangGraph 串成
`plan -> execute -> review -> (replan)` 流程。训练用 LLaMA-Factory 做 QLoRA，
评测统计 JSON 可解析率、schema 合规率、工具名准确率和参数完全匹配率。

## 目录

```
mas/
├── configs/                 # QLoRA 训练、导出、推理配置
├── data/                    # 数据格式、样例、生成脚本输出
├── notebooks/               # Kaggle notebook
├── report/                  # Overleaf 用的 ctexart 模板
├── scripts/                 # 环境、数据、训练、部署
├── src/
│   ├── agents/              # planner / executor / reviewer
│   ├── graph/               # LangGraph 工作流
│   ├── llm/                 # OpenAI 与本地 HF 两种后端
│   ├── tools/               # 工具注册表
│   └── eval/                # 指标与评测入口
└── tests/
```

## 本地环境

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

Windows 上如果已经 clone 过，直接进目录建虚拟环境也可以：

```powershell
cd E:\学校\mas
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

跑测试：

```bash
pytest -q
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

## AutoDL（Kaggle 验证不过时）

Kaggle 手机验证对国内号码不太稳定。备选是 AutoDL 的 4090：

```bash
# AutoDL 实例：4090 24GB，PyTorch 2.5，数据盘 50GB
cd /root/autodl-tmp
git clone git@github.com:starthong/mas.git
cd mas
bash scripts/setup_autodl.sh
```

先小跑 50 条，确认数据、模型和显存都没问题：

```bash
SMOKE=1 bash scripts/train_lora.sh
```

再全量训练：

```bash
bash scripts/train_lora.sh
```

3B 在 4090 上大约 30–60 分钟，费用通常不到 10 元。训练输出在
`/root/autodl-tmp/output/mas-qwen25-3b-toolcall-lora`。

评测：

```bash
bash scripts/eval_autodl.sh ft
bash scripts/eval_autodl.sh base
```

需要起 OpenAI 兼容服务时，再装 vLLM：

```bash
bash scripts/install_vllm.sh
bash scripts/serve_vllm.sh
```

vLLM 可能升级 torch，所以放在训练完成之后装。默认服务 3B base 和
`toolcall` 两个模型名，端口 8000。

## 数据

```bash
python scripts/build_toolcall_dataset.py
```

默认生成约 700 条训练样本和 40 条测试样本，覆盖天气、计算器、搜索和
无需工具四类。模板在脚本里，改起来比较直接。正式实验前建议扩到 1000 条以上，
并按工具划分训练/测试，避免模板泄漏。

`data/samples/toolcall_sample.jsonl` 只是三条格式样例，不要拿它直接训练。

## 评测

AutoDL 上直接用包装脚本：

```bash
bash scripts/eval_autodl.sh ft
bash scripts/eval_autodl.sh base
```

其它环境用 `--backend` 指定后端：

```bash
# OpenAI 兼容接口（vLLM）
python -m src.eval.run_eval --backend openai --model toolcall \
  --data data/test/toolcall_test.jsonl --out results/ft.json

# 本地 transformers + peft
python -m src.eval.run_eval --backend hf \
  --base-model /kaggle/temp/models/Qwen2.5-3B-Instruct \
  --adapter /kaggle/working/output/qwen25-3b-toolcall-lora \
  --data data/test/toolcall_test.jsonl --out results/ft.json --limit 100
```

base 模型去掉 `--adapter` 再跑一遍，得到 `results/base.json`。

## 报告

`report/main.tex` 用 `ctexart`，在 Overleaf 里编译：

1. New Project -> Blank Project
2. 上传 `main.tex`、`refs.bib`、`figures/`
3. Menu -> Compiler -> **XeLaTeX**

## 备注

- `search` 目前是 mock，系统级实验前要换成本地语料 + BM25 或 embedding。
- Executor 现在一次只调用一个工具，多步 ReAct 还没写。
- 训练数据是模板合成的，适合验证格式，不足以证明真实任务能力。
- 报告里需要写清楚：协作模式对比、LoRA 前后对比、稳定性指标，三者缺一不可。
