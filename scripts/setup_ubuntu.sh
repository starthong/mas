#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  echo "[setup] installing git, curl and basic tools"
  sudo apt-get update
  sudo apt-get install -y git curl build-essential openssh-server
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] installing Python 3.11"
uv python install 3.11

if [ ! -x ".venv/bin/python" ]; then
  echo "[setup] creating .venv"
  rm -rf .venv
  uv venv --python 3.11 .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] installing Python dependencies"
uv pip install -r requirements.txt

cat <<'EOF'

[done] Environment is ready.

Next:
  source .venv/bin/activate
  cp .env.example .env
  python -m src.main "北京今天天气怎么样？"

Git identity (once per machine):
  git config --global user.name  "Your Name"
  git config --global user.email "you@example.com"
EOF
