from __future__ import annotations

import ast
import operator

from src.tools.registry import Tool, ToolRegistry
from src.tools.web import get_weather, search

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _eval_ast(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """安全求值小型算术表达式。"""
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("expression must be a string shorter than 200 chars")
    tree = ast.parse(expression, mode="eval")
    return str(_eval_ast(tree))


def final_answer(answer: str) -> str:
    return answer


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="search",
            description="Search a local knowledge base for relevant documents.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            func=search,
        )
    )

    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. (23*17+9)/5.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            func=calculator,
        )
    )

    registry.register(
        Tool(
            name="get_weather",
            description="Get the current weather for a city. Currently a mock.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name in Chinese."}
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            func=get_weather,
        )
    )

    registry.register(
        Tool(
            name="final_answer",
            description="Return the final answer when no external tool is needed.",
            parameters={
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Final answer text."}
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
            func=final_answer,
        )
    )

    return registry
