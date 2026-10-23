#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
BASE_MODEL="${BASE_MODEL:-/kaggle/temp/models/Qwen2.5-3B-Instruct}"
ADAPTER="${ADAPTER:-/kaggle/input/datasets/starthong/mas-3b-toolcall-output}"
DATA="${DATA:-data/test/toolcall_hard.jsonl}"
OUT_DIR="${OUT_DIR:-results}"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"

echo "[1/4] base, temperature 0"
python -m src.eval.run_eval \
  --backend hf \
  --base-model "$BASE_MODEL" \
  --data "$DATA" \
  --out "$OUT_DIR/hard-base-0.json"

echo "[2/4] LoRA, temperature 0"
python -m src.eval.run_eval \
  --backend hf \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --data "$DATA" \
  --out "$OUT_DIR/hard-ft-0.json"

echo "[3/4] base, temperature 0.7"
python -m src.eval.run_eval \
  --backend hf \
  --base-model "$BASE_MODEL" \
  --data "$DATA" \
  --temperature 0.7 \
  --out "$OUT_DIR/hard-base-07.json"

echo "[4/4] LoRA, temperature 0.7"
python -m src.eval.run_eval \
  --backend hf \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --data "$DATA" \
  --temperature 0.7 \
  --out "$OUT_DIR/hard-ft-07.json"

echo "[done] $OUT_DIR/hard-*.json"
