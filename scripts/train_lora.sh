#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/mas}"
LF_DIR="${LF_DIR:-/root/LLaMA-Factory}"
CONFIG="${1:-$REPO_DIR/configs/train_qlora.yaml}"

cd "$LF_DIR"

CMD=(llamafactory-cli train "$CONFIG")
if [ "${SMOKE:-0}" = "1" ]; then
  echo "[train] smoke mode: 50 samples, 20 steps"
  CMD+=(
    max_samples=50
    max_steps=20
    output_dir=/root/autodl-tmp/output/smoke
  )
fi

"${CMD[@]}"
