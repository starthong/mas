#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"

python -m pip install -U modelscope

mkdir -p "$MODEL_DIR"
modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"

echo "[done] model saved to $MODEL_DIR"
