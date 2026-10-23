from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.llm.parsing import parse_json_loose
from src.llm.prompts import SUPERVISOR_SYSTEM

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class Supervisor:
    """Choose the next action after each executor step."""

    def __init__(self, client: LLMClient, temperature: float = 0.0) -> None:
        self.client = client
        self.temperature = temperature
        self.last_error: str | None = None

    def decide(
        self,
        task: str,
        plan: list[dict[str, Any]],
        results: list[dict[str, Any]],
        cursor: int,
        rounds: int,
    ) -> dict[str, Any]:
        self.last_error = None
        user = (
            f"Task:\n{task}\n\n"
            f"Plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"Results so far:\n{json.dumps(results, ensure_ascii=False, indent=2)}\n\n"
            f"Next unexecuted subtask index: {cursor}\n"
            f"Review rounds so far: {rounds}\n\n"
            "Decide the next action. Return only the JSON object."
        )
        raw = ""
        try:
            raw = self.client.chat(
                [
                    {"role": "system", "content": SUPERVISOR_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=256,
            )
            data = parse_json_loose(raw)
            action = str(data.get("action", "")).strip().lower()
            if action not in ("execute", "review", "finish"):
                raise ValueError(f"unknown supervisor action: {action!r}")

            if action == "execute":
                if cursor >= len(plan):
                    raise ValueError("supervisor asked to execute with no subtasks left")
                subtask_id = data.get("subtask_id", plan[cursor].get("id"))
                return {
                    "action": "execute",
                    "subtask_id": subtask_id,
                    "reason": str(data.get("reason", "")),
                }
            return {
                "action": action,
                "subtask_id": None,
                "reason": str(data.get("reason", "")),
            }
        except Exception as exc:  # noqa: BLE001 - fall back to a safe schedule
            self.last_error = f"{type(exc).__name__}: {exc}"
            if cursor < len(plan):
                return {
                    "action": "execute",
                    "subtask_id": plan[cursor].get("id"),
                    "reason": "fallback: execute next subtask",
                }
            return {
                "action": "review",
                "subtask_id": None,
                "reason": "fallback: review completed plan",
            }
