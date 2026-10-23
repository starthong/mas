from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.llm.parsing import parse_json_loose
from src.llm.prompts import PLANNER_SYSTEM, PLANNER_SYSTEM_FEWSHOT

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class Planner:
    """Decompose a task into ordered subtasks."""

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        few_shot: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            PLANNER_SYSTEM_FEWSHOT if few_shot else PLANNER_SYSTEM
        )
        self.last_error: str | None = None

    def plan(self, task: str, critique: str = "") -> list[dict[str, Any]]:
        self.last_error = None
        user = f"Task:\n{task}\n"
        if critique:
            user += f"\nThe reviewer rejected the previous attempt:\n{critique}\n"
            user += "Produce a revised plan that fixes these issues.\n"

        raw = self.client.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=512,
        )
        try:
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
        except Exception as exc:
            # A broken plan should not abort the whole experiment.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return [{"id": 1, "goal": task, "expected_output": ""}]
