#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
BASE_MODEL="${BASE_MODEL:-/kaggle/temp/models/Qwen2.5-3B-Instruct}"
ADAPTER="${ADAPTER:-/kaggle/input/datasets/starthong/mas-3b-toolcall-output}"
OUT_DIR="${OUT_DIR:-results/postfix}"
TASKS="${TASKS:-data/system_tasks.jsonl}"
PLANNER_ADAPTER="${PLANNER_ADAPTER:-}"
REVIEWER_ADAPTER="${REVIEWER_ADAPTER:-}"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"

role_args=()
if [ -n "$PLANNER_ADAPTER" ]; then
  role_args+=(--planner-adapter "$PLANNER_ADAPTER")
fi
if [ -n "$REVIEWER_ADAPTER" ]; then
  role_args+=(--reviewer-adapter "$REVIEWER_ADAPTER")
fi

echo "===== base 31: single + sequential ====="
python -m src.experiments.run_modes \
  --backend hf \
  --base-model "$BASE_MODEL" \
  --tasks "$TASKS" \
  --modes single,sequential \
  --planner base \
  --reviewer base \
  --runs 1 \
  --temperatures 0 \
  --out "$OUT_DIR/base31.json"

echo "===== LoRA 31: single + sequential ====="
python -m src.experiments.run_modes \
  --backend hf \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --tasks "$TASKS" \
  --modes single,sequential \
  --planner lora \
  --reviewer lora \
  "${role_args[@]}" \
  --runs 1 \
  --temperatures 0 \
  --out "$OUT_DIR/lora31.json"

for spec in \
  "all_lora:lora:lora" \
  "lora_planner_base_reviewer:lora:base" \
  "no_reviewer:lora:none" \
  "base_planner_lora_reviewer:base:lora" \
  "base_both:base:base"; do
  name="${spec%%:*}"
  rest="${spec#*:}"
  planner="${rest%%:*}"
  reviewer="${rest##*:}"
  echo "===== role ${name} ====="
  python -m src.experiments.run_modes \
    --backend hf \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --tasks "$TASKS" \
    --modes sequential \
    --planner "$planner" \
    --reviewer "$reviewer" \
    "${role_args[@]}" \
    --runs 1 \
    --temperatures 0 \
    --out "$OUT_DIR/role-${name}.json"
done

echo "===== temperature stability (3 runs, 0 and 0.7) ====="
python -m src.experiments.run_modes \
  --backend hf \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --tasks "$TASKS" \
  --modes sequential \
  --planner lora \
  --reviewer lora \
  --planner-examples \
  "${role_args[@]}" \
  --runs 3 \
  --temperatures 0,0.7 \
  --out "$OUT_DIR/temperature.json"

echo "===== max_rounds on 14 multi tasks ====="
for rounds in 1 2 3; do
  python -m src.experiments.run_modes \
    --backend hf \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --tasks data/system_tasks_expanded.jsonl \
    --categories multi \
    --modes sequential \
    --planner lora \
    --reviewer lora \
    --planner-examples \
    "${role_args[@]}" \
    --max-rounds "$rounds" \
    --runs 1 \
    --temperatures 0 \
    --out "$OUT_DIR/sensitivity-r${rounds}.json"
done

echo "===== done: $OUT_DIR ====="
