#!/usr/bin/env python
"""Minimal smoke test for an OpenAI-compatible endpoint.

Run from the repository root:
    python scripts/smoke_test.py
    MODEL_NAME=base python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.llm.client import LLMClient  # noqa: E402
from src.llm.prompts import build_system_prompt  # noqa: E402
from src.tools.builtin import build_default_registry  # noqa: E402


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    registry = build_default_registry()
    client = LLMClient()
    model = os.getenv("MODEL_NAME") or os.getenv("FT_MODEL", "toolcall")

    task = "北京今天天气怎么样？"
    messages = [
        {"role": "system", "content": build_system_prompt(registry.describe())},
        {"role": "user", "content": task},
    ]

    raw = client.chat(messages, model=model, temperature=0.0, max_tokens=256)
    print("model:", model)
    print("raw:", raw)

    try:
        call = json.loads(raw)
    except json.JSONDecodeError as exc:
        print("JSON parse failed:", exc)
        return 1

    print("parsed:", json.dumps(call, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
