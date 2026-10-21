#!/usr/bin/env bash
set -euo pipefail

# vLLM can upgrade torch. Install it only after QLoRA training is finished.
pip install -U vllm
vllm --version
