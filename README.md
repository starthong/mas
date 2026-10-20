# mas

GOODLAB 挑战：基于开源多智能体框架的协作系统，以及面向工具调用 JSON 格式的 LoRA 微调实验。

当前仓库是**可运行的骨架**：包含三角色 Agent、工具注册表、LangGraph 工作流、训练配置、评测指标和 Overleaf 报告模板。复杂逻辑（多步 ReAct、真实检索、完整实验脚本）在代码中以 `TODO` 标出，按周计划逐步补齐。

## 1. 架构

```text
                 ┌─────────────┐
   task ────────▶│   Planner   │  拆解子任务，输出 JSON plan
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Executor   │  选择工具，输出 JSON tool call
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Reviewer   │  审查结果，输出 pass / issues
                 └──────┬──────┘
                        │
          pass=false 且未超轮数
                        │
                        └──────▶ Planner（带 critique 重规划）
```

- 模型通过 OpenAI 兼容接口访问：本地开发连 AutoDL vLLM，或任何兼容服务。
- `base` 是基座模型，`toolcall` 是加载 LoRA adapter 后的模型；同一套代码切换模型名即可做对比。
- 默认工具：`search`（mock）、`calculator`（安全 AST 求值）、`get_weather`（mock）、`final_answer`。

## 2. 目录

```text
mas/
├── README.md
├── .gitignore
├── .gitattributes
├── .env.example
├── requirements.txt
├── configs/
│   ├── train_qlora.yaml
│   ├── export_lora.yaml
│   └── inference.yaml
├── data/
│   ├── README.md
│   ├── dataset_info.json
│   ├── samples/toolcall_sample.jsonl
│   └── test/README.md
├── results/
├── submission/
├── scripts/
│   ├── setup_ubuntu.sh
│   ├── download_model.sh
│   ├── train_lora.sh
│   ├── serve_vllm.sh
│   └── smoke_test.py
├── src/
│   ├── main.py
│   ├── llm/{client,prompts,parsing}.py
│   ├── tools/{registry,builtin}.py
│   ├── agents/{planner,executor,reviewer}.py
│   ├── graph/workflow.py
│   └── eval/{metrics,run_eval}.py
├── tests/test_metrics.py
└── report/
    ├── main.tex
    ├── refs.bib
    └── figures/
```

## 3. 本地环境（Ubuntu）

### 如果代码先在 Windows 上

本骨架可能先在 Windows 工作区生成。Windows 已装 git 时，可以直接在 Windows 初始化并推送，再在 Ubuntu clone：

```powershell
cd "D:\path\to\mas"
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱"
git init
git add .
git commit -m "chore: init mas skeleton"
git branch -M main
git remote add origin https://github.com/<你的用户名>/mas.git
git push -u origin main
```

然后在 Ubuntu 里：

```bash
sudo apt update && sudo apt install -y git
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱"
cd /data
git clone https://github.com/<你的用户名>/mas.git
```

私有仓库用 HTTPS 克隆时需要 GitHub PAT；更稳的是在 Ubuntu 生成 SSH key，把公钥加到 GitHub 后改用 `git@github.com:<用户名>/mas.git`。

### 直接在 Ubuntu 初始化

Ubuntu 默认没有 git，先装：

```bash
sudo apt update
sudo apt install -y git curl build-essential openssh-server
git --version
```

配置身份：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

安装 Python 3.11 和依赖：

```bash
cd /data/mas
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

脚本会安装 `uv`、Python 3.11，并创建 `.venv`、安装 `requirements.txt`。它不会安装 TeX Live，报告在 Overleaf 编译。

## 4. 启动模型服务

训练和部署在 AutoDL 上完成，详见 `configs/` 和 `scripts/`。最简流程：

```bash
# AutoDL 上
bash scripts/download_model.sh
bash scripts/train_lora.sh
bash scripts/serve_vllm.sh
```

本地 Windows 开 SSH 隧道：

```bash
ssh -p <AutoDL端口> -L 8000:127.0.0.1:8000 root@<AutoDL地址>
```

复制环境变量：

```bash
cp .env.example .env
# 默认 OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

检查模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

应同时看到 `base` 和 `toolcall`。

## 5. 运行

```bash
source .venv/bin/activate
python -m src.main "北京今天天气怎么样？"
```

指定模型做对比：

```bash
MODEL_NAME=base python -m src.main "计算 (23*17+9)/5"
MODEL_NAME=toolcall python -m src.main "计算 (23*17+9)/5"
```

输出包含 `plan`、`results`、`review` 和简化 trace。

## 6. 评测

准备测试集 JSONL，每行至少包含 `messages`（最后一条是 assistant 参考答案）：

```bash
python -m src.eval.run_eval \
  --data data/test/toolcall_test.jsonl \
  --model toolcall \
  --out results/toolcall_eval.json
```

会输出 JSON 可解析率、schema 合规率、工具名准确率、参数完全匹配率等。格式错误分类见 `src/eval/metrics.py`。

## 7. 训练

数据格式和构造建议见 `data/README.md`。训练配置在 `configs/train_qlora.yaml`，默认：

- Qwen2.5-7B-Instruct
- QLoRA，rank 16，alpha 32
- 1k 样本，2 epoch，bf16，gradient checkpointing

先小跑验证链路：

```bash
cd /root/LLaMA-Factory
llamafactory-cli train /root/autodl-tmp/mas/configs/train_qlora.yaml \
  max_samples=50 max_steps=20 \
  output_dir=/root/autodl-tmp/output/smoke
```

再全量训练：

```bash
bash /root/autodl-tmp/mas/scripts/train_lora.sh
```

## 8. 报告

`report/main.tex` 是 `ctexart` 模板，直接用 Overleaf 打开：

1. New Project → Blank Project；
2. 上传 `main.tex`、`refs.bib`、`figures/`；
3. Menu → Compiler → **XeLaTeX**；
4. 编译。

报告源文件以 GitHub 为准；Overleaf 只负责编译。免费版没有 Git 集成时，下载 Source zip 覆盖本地 `report/` 后提交。

## 9. 当前状态

- [x] 三角色 Agent 骨架
- [x] LangGraph 工作流与重试
- [x] 工具注册表与三个示例工具
- [x] 训练/导出/部署配置
- [x] 格式评测指标
- [ ] 多步 ReAct / 并行工具调用
- [ ] 真实检索工具与固定语料
- [ ] 协作模式消融脚本
- [ ] 30 条系统评测任务
- [ ] 完整实验数据与图表

## 10. 提交前检查

- [ ] `README.md` 的复现命令能在干净环境跑通
- [ ] 不提交模型权重、checkpoint、原始大数据
- [ ] `report/main.pdf` 用 XeLaTeX 编译无错误
- [ ] 报告包含协作模式对比、LoRA 前后对比、稳定性分析
- [ ] 打 tag：`git tag submission && git push --tags`
