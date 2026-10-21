from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.llm.parsing import parse_json_loose
from src.llm.prompts import build_system_prompt
from src.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class Executor:
    """Execute one subtask by selecting and calling one tool.

    This is intentionally a single-step executor. The next iteration should
    turn it into a ReAct loop with a max step budget and per-step observations.
    """

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.registry = registry
        self.temperature = temperature

    def run(
        self,
        task: str,
        subtask: dict[str, Any],
        context: str = "",
    ) -> dict[str, Any]:
        system = build_system_prompt(self.registry.describe())
        user = (
            f"Overall task:\n{task}\n\n"
            f"Current subtask:\n{json.dumps(subtask, ensure_ascii=False)}\n\n"
            f"Previous results:\n{context or '(none)'}\n\n"
            "Choose one tool call for the current subtask. "
            "Return only the JSON object."
        )

        raw = ""
        try:
            raw = self.client.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=512,
            )
            call = parse_json_loose(raw)
            name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(name, str) or not name:
                raise ValueError("tool call has no name")
            if not isinstance(arguments, dict):
                raise ValueError("tool call arguments must be an object")

            result = self.registry.call(name, arguments)
            return {
                "subtask_id": subtask.get("id"),
                "goal": subtask.get("goal", ""),
                "tool": name,
                "arguments": arguments,
                "result": result,
                "error": None,
                "raw": raw,
            }
        except Exception as exc:  # noqa: BLE001 - surface model/tool failures to review
            return {
                "subtask_id": subtask.get("id"),
                "goal": subtask.get("goal", ""),
                "tool": None,
                "arguments": {},
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "raw": raw,
            }


def results_to_context(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in results:
        if item.get("error"):
            lines.append(
                f"- subtask {item.get('subtask_id')}: ERROR {item['error']}"
            )
        else:
            lines.append(
                f"- subtask {item.get('subtask_id')} -> "
                f"{item.get('tool')}({json.dumps(item.get('arguments', {}), ensure_ascii=False)}) "
                f"= {item.get('result')}"
            )
    return "\n".join(lines) if lines else "(none)"
