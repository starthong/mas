#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/mas}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"

cd /root
if [ ! -d "LLaMA-Factory" ]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e .
pip install -U modelscope bitsandbytes

pip install -r "$REPO_DIR/requirements.txt"

MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
  bash "$REPO_DIR/scripts/download_model.sh" "$MODEL_DIR"

command -v llamafactory-cli
echo "[done] LLaMA-Factory, repo deps and 3B model are ready"
