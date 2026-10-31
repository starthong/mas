from __future__ import annotations

import json
from typing import Any


def parse_json_strict(text: str) -> dict[str, Any]:
    """严格解析模型原始输出。

    评测使用该函数：Markdown 围栏、解释文字或尾随文本都算格式错误，
    因此这里故意不做任何修复。
    """
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj


def parse_json_loose(text: str) -> dict[str, Any]:
    """运行时尽力解析，不能用于报告指标。

    即使模型把 JSON 包在 Markdown 中或加了解释，Agent 也应继续运行；
    评测流程使用 ``parse_json_strict``。
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
        candidate = raw[start : end + 1] if start != -1 and end > start else raw
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            # 小模型偶尔会漏逗号或右括号；运行时修复，严格评测仍使用原始输出。
            from json_repair import repair_json

            repaired = repair_json(raw)
            obj = json.loads(repaired)

    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj
