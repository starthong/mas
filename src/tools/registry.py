from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


class ToolError(RuntimeError):
    """Raised when a tool is unknown or its arguments are invalid."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names()]

    def describe(self) -> str:
        return json.dumps(self.schemas(), ensure_ascii=False, indent=2)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be a JSON object")
        tool = self.get(name)
        try:
            return tool.func(**arguments)
        except TypeError as exc:
            raise ToolError(f"invalid arguments for {name}: {exc}") from exc
