#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
BASE_MODEL="${BASE_MODEL:-/kaggle/temp/models/Qwen2.5-3B-Instruct}"
ADAPTER="${ADAPTER:-/kaggle/input/datasets/starthong/mas-3b-toolcall-output}"

cd "$REPO_DIR"
for rounds in 1 2 3; do
  python -m src.experiments.run_modes \
    --backend hf \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --tasks data/system_tasks_v2.jsonl \
    --categories multi \
    --modes sequential \
    --planner lora \
    --reviewer lora \
    --planner-examples \
    --max-rounds "$rounds" \
    --runs 1 \
    --temperatures 0 \
    --out "results/sensitivity-r${rounds}.json"
done
