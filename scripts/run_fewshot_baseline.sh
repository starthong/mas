#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
BASE_MODEL="${BASE_MODEL:-/kaggle/temp/models/Qwen2.5-3B-Instruct}"
OUT_DIR="${OUT_DIR:-results}"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"

for spec in "hard:default" "ood:ood"; do
  dataset="${spec%%:*}"
  registry="${spec##*:}"
  for temp in 0 0.7; do
    suffix="${temp/./}"
    echo "[few-shot] $dataset temp=$temp"
    python -m src.eval.run_eval \
      --backend hf \
      --registry "$registry" \
      --base-model "$BASE_MODEL" \
      --data "data/test/toolcall_${dataset}_fewshot.jsonl" \
      --temperature "$temp" \
      --out "$OUT_DIR/${dataset}-base-fewshot-${suffix}.json"
  done
done
