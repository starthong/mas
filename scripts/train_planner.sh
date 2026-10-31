#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/mas}"
cd "$REPO_DIR"
llamafactory-cli train configs/train_planner_lora.yaml
