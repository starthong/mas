#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
ADAPTER="${ADAPTER:-/root/autodl-tmp/output/mas-qwen25-3b-toolcall-lora}"
PORT="${PORT:-8000}"

vllm serve "$BASE_MODEL" \
  --served-model-name base \
  --enable-lora \
  --lora-modules "toolcall=$ADAPTER" \
  --max-lora-rank 16 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port "$PORT"
