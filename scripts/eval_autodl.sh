#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MODE="${1:-ft}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
ADAPTER="${ADAPTER:-/root/autodl-tmp/output/mas-qwen25-3b-toolcall-lora}"
DATA="${DATA:-data/test/toolcall_test.jsonl}"
LIMIT="${LIMIT:-100}"
OUT="${OUT:-results/${MODE}.json}"

ARGS=(
  --backend hf
  --base-model "$BASE_MODEL"
  --data "$DATA"
  --out "$OUT"
  --limit "$LIMIT"
)

if [ "$MODE" = "ft" ]; then
  ARGS+=(--adapter "$ADAPTER")
elif [ "$MODE" != "base" ]; then
  echo "usage: bash scripts/eval_autodl.sh [ft|base]" >&2
  exit 2
fi

python -m src.eval.run_eval "${ARGS[@]}"
