from __future__ import annotations

from src.eval.metrics import (
    classify_format_error,
    exact_match,
    json_parse_ok,
    validate_call,
    wilson_ci,
)

TOOLS = [
    {
        "name": "calculator",
        "description": "calc",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    }
]


def test_json_parse_ok() -> None:
    assert json_parse_ok('{"name": "calculator", "arguments": {"expression": "1+1"}}')
    assert not json_parse_ok('```json\n{"name": "x"}\n```')
    assert not json_parse_ok("not json")


def test_validate_call() -> None:
    good = {"name": "calculator", "arguments": {"expression": "1+1"}}
    bad_tool = {"name": "unknown", "arguments": {}}
    bad_args = {"name": "calculator", "arguments": {"expression": 1}}
    assert validate_call(good, TOOLS)
    assert not validate_call(bad_tool, TOOLS)
    assert not validate_call(bad_args, TOOLS)


def test_classify_format_error() -> None:
    assert classify_format_error('{"name": "calculator", "arguments": {}}') == "ok"
    assert classify_format_error("```json\n{}\n```") == "markdown_fence"
    assert classify_format_error("Here is the call: {}") == "extra_text"
    assert classify_format_error("") == "empty"


def test_exact_match() -> None:
    ref = {"name": "calculator", "arguments": {"expression": "1+1"}}
    assert exact_match(ref, ref)
    assert not exact_match({"name": "calculator", "arguments": {"expression": "2"}}, ref)


def test_wilson_ci() -> None:
    low, high = wilson_ci(50, 100)
    assert low < 0.5 < high
