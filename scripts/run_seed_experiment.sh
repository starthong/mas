#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
BASE_MODEL="${BASE_MODEL:-/kaggle/temp/models/Qwen2.5-3B-Instruct}"
SEEDS="${SEEDS:-42,43,44}"
DATASETS="${DATASETS:-v1,v2}"
OUT_DIR="${OUT_DIR:-results/seeds}"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"

for dataset in ${DATASETS//,/ }; do
  for seed in ${SEEDS//,/ }; do
    config="configs/seeds/${dataset}-seed${seed}.yaml"
    adapter="/kaggle/working/output/seeds/${dataset}-seed${seed}"
    echo "===== train ${dataset} seed=${seed} ====="
    llamafactory-cli train "$config"

    echo "===== eval ${dataset} seed=${seed} ====="
    python -m src.eval.run_eval \
      --backend hf \
      --base-model "$BASE_MODEL" \
      --adapter "$adapter" \
      --data data/test/toolcall_hard.jsonl \
      --out "$OUT_DIR/${dataset}-seed${seed}-hard.json"

    python -m src.eval.run_eval \
      --backend hf \
      --registry ood \
      --base-model "$BASE_MODEL" \
      --adapter "$adapter" \
      --data data/test/toolcall_ood.jsonl \
      --out "$OUT_DIR/${dataset}-seed${seed}-ood.json"
  done
done

echo "done: $OUT_DIR"
