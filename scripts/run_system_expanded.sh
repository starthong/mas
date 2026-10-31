#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
BASE_MODEL="${BASE_MODEL:-/kaggle/temp/models/Qwen2.5-3B-Instruct}"
ADAPTER="${ADAPTER:-/kaggle/input/datasets/starthong/mas-3b-toolcall-output}"
MODES="${MODES:-sequential,supervisor,debate}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
OUT="${OUT:-results/system-expanded.json}"

# 默认使用主 adapter（v1 数据训练），不是 v2 扩展数据 adapter。
cd "$REPO_DIR"
python -m src.experiments.run_modes \
  --backend hf \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --tasks data/system_tasks_expanded.jsonl \
  --modes "$MODES" \
  --planner lora \
  --reviewer lora \
  --planner-examples \
  --max-rounds "$MAX_ROUNDS" \
  --runs 1 \
  --temperatures 0 \
  --out "$OUT"
