from __future__ import annotations

import json
from typing import Any


def parse_json_strict(text: str) -> dict[str, Any]:
    """Parse exactly what the model returned.

    Used by evaluation: markdown fences, prose, or trailing text must count as
    format errors, so this function deliberately does not repair anything.
    """
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj


def parse_json_loose(text: str) -> dict[str, Any]:
    """Best-effort parser used at runtime, not for reporting metrics.

    The agent should keep running even if the model wraps JSON in a markdown
    fence or adds a short explanation. The evaluation pipeline uses
    ``parse_json_strict`` instead.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model output")

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        obj = json.loads(raw[start : end + 1])

    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj
