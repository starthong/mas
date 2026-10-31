#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
BASE_MODEL="${BASE_MODEL:-/kaggle/temp/models/Qwen2.5-3B-Instruct}"
ADAPTER="${ADAPTER:-/kaggle/input/datasets/starthong/mas-3b-toolcall-output}"
OUT_DIR="${OUT_DIR:-results/constrained}"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"

for split in hard ood; do
  registry="default"
  data="data/test/toolcall_hard.jsonl"
  if [ "$split" = "ood" ]; then
    registry="ood"
    data="data/test/toolcall_ood.jsonl"
  fi

  python scripts/run_constrained_vllm.py \
    --data "$data" \
    --registry "$registry" \
    --base-model "$BASE_MODEL" \
    --out "$OUT_DIR/${split}-base.json"

  python scripts/run_constrained_vllm.py \
    --data "$data" \
    --registry "$registry" \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --out "$OUT_DIR/${split}-lora.json"
done
