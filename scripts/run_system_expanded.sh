#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
BASE_MODEL="${BASE_MODEL:-/kaggle/temp/models/Qwen2.5-3B-Instruct}"
ADAPTER="${ADAPTER:-/kaggle/input/datasets/starthong/mas-3b-toolcall-output}"
MODES="${MODES:-sequential,supervisor,debate}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
OUT="${OUT:-results/system-expanded.json}"
CATEGORIES="${CATEGORIES:-}"
PLANNER_SAMPLES="${PLANNER_SAMPLES:-1}"
PLANNER_ADAPTER="${PLANNER_ADAPTER:-}"
REVIEWER_ADAPTER="${REVIEWER_ADAPTER:-}"
PLANNER_MODE="${PLANNER_MODE:-lora}"
PLANNER_BASE_MODEL="${PLANNER_BASE_MODEL:-}"

# 默认使用主 adapter（v1 数据训练），不是 v2 扩展数据 adapter。
cd "$REPO_DIR"
extra=()
if [ -n "$CATEGORIES" ]; then
  extra+=(--categories "$CATEGORIES")
fi
if [ -n "$PLANNER_ADAPTER" ]; then
  extra+=(--planner-adapter "$PLANNER_ADAPTER")
fi
if [ -n "$REVIEWER_ADAPTER" ]; then
  extra+=(--reviewer-adapter "$REVIEWER_ADAPTER")
fi
if [ -n "$PLANNER_BASE_MODEL" ]; then
  extra+=(--planner-base-model "$PLANNER_BASE_MODEL")
fi
python -m src.experiments.run_modes \
  --backend hf \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --tasks data/system_tasks_expanded.jsonl \
  --modes "$MODES" \
  --planner "$PLANNER_MODE" \
  --reviewer lora \
  --planner-examples \
  --planner-samples "$PLANNER_SAMPLES" \
  --max-rounds "$MAX_ROUNDS" \
  --runs 1 \
  --temperatures 0 \
  "${extra[@]}" \
  --out "$OUT"
