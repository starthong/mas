from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.llm.parsing import parse_json_loose
from src.llm.prompts import PLANNER_SYSTEM

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class Planner:
    """Decompose a task into ordered subtasks."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def plan(self, task: str, critique: str = "") -> list[dict[str, Any]]:
        user = f"Task:\n{task}\n"
        if critique:
            user += f"\nThe reviewer rejected the previous attempt:\n{critique}\n"
            user += "Produce a revised plan that fixes these issues.\n"

        raw = self.client.chat(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        data = parse_json_loose(raw)
        subtasks = data.get("subtasks")
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError(f"planner returned no subtasks: {raw!r}")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(subtasks[:5], start=1):
            if not isinstance(item, dict):
                raise ValueError(f"subtask is not an object: {item!r}")
            goal = str(item.get("goal", "")).strip()
            if not goal:
                raise ValueError(f"subtask has no goal: {item!r}")
            normalized.append(
                {
                    "id": item.get("id", index),
                    "goal": goal,
                    "expected_output": str(item.get("expected_output", "")).strip(),
                }
            )
        return normalized
