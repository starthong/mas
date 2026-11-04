from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from jsonschema import Draft202012Validator


def json_parse_ok(text: str) -> bool:
    try:
        obj = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(obj, dict)


def validate_call(call: Any, tools: list[dict[str, Any]] | dict[str, Any]) -> bool:
    """只有工具名存在且参数符合 schema 时才返回 True。"""
    if not isinstance(call, dict):
        return False

    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return False

    if isinstance(tools, dict):
        tool_map = tools
    else:
        tool_map = {item.get("name"): item for item in tools if isinstance(item, dict)}

    tool = tool_map.get(name)
    if not isinstance(tool, dict):
        return False

    schema = tool.get("parameters")
    if not isinstance(schema, dict):
        return False

    try:
        Draft202012Validator(schema).validate(arguments)
    except Exception:  # noqa: BLE001 - any validation failure is a failed call
        return False
    return True


def tool_name_match(pred: Any, ref: Any) -> bool:
    return (
        isinstance(pred, dict)
        and isinstance(ref, dict)
        and pred.get("name") == ref.get("name")
    )


_LANGUAGE_ALIASES = {
    "en": "English",
    "english": "English",
    "fr": "French",
    "french": "French",
    "ja": "Japanese",
    "japanese": "Japanese",
    "zh": "Chinese",
    "chinese": "Chinese",
    "de": "German",
    "german": "German",
    "es": "Spanish",
    "spanish": "Spanish",
    "ko": "Korean",
    "korean": "Korean",
    "ru": "Russian",
    "russian": "Russian",
}


_STOCK_ALIASES = {
    "腾讯": "0700.HK",
    "腾讯控股": "0700.HK",
    "tencent": "0700.HK",
    "特斯拉": "TSLA",
    "tesla": "TSLA",
    "阿里巴巴": "BABA",
    "alibaba": "BABA",
    "亚马逊": "AMZN",
    "amazon": "AMZN",
    "英伟达": "NVDA",
    "nvidia": "NVDA",
    "苹果": "AAPL",
    "apple": "AAPL",
    "微软": "MSFT",
    "microsoft": "MSFT",
    "谷歌": "GOOGL",
    "google": "GOOGL",
}


def normalize_call(call: Any) -> Any:
    """把 OOD 中等价的枚举/别名写法归一化，避免 en/English、公司名/代码被 exact match 判错。"""
    if not isinstance(call, dict):
        return call
    normalized = json.loads(json.dumps(call, ensure_ascii=False))
    if normalized.get("name") == "translate":
        arguments = normalized.get("arguments")
        if isinstance(arguments, dict):
            language = arguments.get("target_language")
            if isinstance(language, str):
                arguments["target_language"] = _LANGUAGE_ALIASES.get(
                    language.lower(), language
                )
    if normalized.get("name") == "get_stock_price":
        arguments = normalized.get("arguments")
        if isinstance(arguments, dict):
            symbol = arguments.get("symbol")
            if isinstance(symbol, str):
                arguments["symbol"] = _STOCK_ALIASES.get(
                    symbol.strip().lower(), symbol
                )
    return normalized


def exact_match(pred: Any, ref: Any) -> bool:
    pred = normalize_call(pred)
    ref = normalize_call(ref)
    return (
        isinstance(pred, dict)
        and isinstance(ref, dict)
        and pred.get("name") == ref.get("name")
        and pred.get("arguments") == ref.get("arguments")
    )


def classify_format_error(text: str) -> str:
    """用于报告错误分析的粗粒度分类。"""
    raw = (text or "").strip()
    if not raw:
        return "empty"
    if raw.startswith("```"):
        return "markdown_fence"
    if "{" not in raw:
        return "no_json"

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        if not raw.startswith("{") or not raw.endswith("}"):
            return "extra_text"
        return "invalid_json"

    if not isinstance(obj, dict):
        return "not_object"
    if "name" not in obj:
        return "missing_name"
    if "arguments" not in obj:
        return "missing_arguments"
    if not isinstance(obj.get("arguments"), dict):
        return "arguments_not_object"
    return "ok"


def error_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(item.get("format_error", "unknown") for item in records))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 95% Wilson 区间。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))
