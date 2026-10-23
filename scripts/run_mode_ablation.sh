#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
BASE_MODEL="${BASE_MODEL:-/kaggle/temp/models/Qwen2.5-3B-Instruct}"
ADAPTER="${ADAPTER:-/kaggle/input/datasets/starthong/mas-3b-toolcall-output}"
MODES="${MODES:-sequential,supervisor,debate}"
RUNS="${RUNS:-1}"
CATEGORIES="${CATEGORIES:-}"
OUT="${OUT:-results/modes-new.json}"

cd "$REPO_DIR"
mkdir -p "$(dirname "$OUT")"

args=(
  --backend hf
  --base-model "$BASE_MODEL"
  --adapter "$ADAPTER"
  --tasks data/system_tasks.jsonl
  --modes "$MODES"
  --planner lora
  --reviewer lora
  --planner-examples
  --runs "$RUNS"
  --temperatures 0
  --out "$OUT"
)
if [ -n "$CATEGORIES" ]; then
  args+=(--categories "$CATEGORIES")
fi

python -m src.experiments.run_modes "${args[@]}"
