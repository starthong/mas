from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.agents.planner import Planner
from src.llm.parsing import parse_json_loose
from src.llm.prompts import JUDGE_SYSTEM, PLANNER_CRITIC_SYSTEM

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class DebatePlanner:
    """Two planners propose plans; a judge picks one."""

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        few_shot: bool = False,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.planner_a = Planner(client, temperature, few_shot=few_shot)
        self.planner_b = Planner(
            client,
            temperature,
            system_prompt=PLANNER_CRITIC_SYSTEM,
        )
        self.last_error: str | None = None
        self.last_choice: int | None = None
        self.last_reason = ""

    def plan(self, task: str, critique: str = "") -> list[dict[str, Any]]:
        self.last_error = None
        plan_a = self.planner_a.plan(task, critique=critique)
        plan_b = self.planner_b.plan(task, critique=critique)

        user = (
            f"Task:\n{task}\n\n"
            f"Plan 1:\n{json.dumps(plan_a, ensure_ascii=False, indent=2)}\n\n"
            f"Plan 2:\n{json.dumps(plan_b, ensure_ascii=False, indent=2)}\n\n"
            "Choose the more complete and executable plan. "
            "Return only the JSON object."
        )
        raw = ""
        try:
            raw = self.client.chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=256,
            )
            data = parse_json_loose(raw)
            choice = int(data.get("choice", 0))
            if choice not in (1, 2):
                raise ValueError(f"judge returned invalid choice: {choice!r}")
            self.last_choice = choice
            self.last_reason = str(data.get("reason", ""))
            return plan_a if choice == 1 else plan_b
        except Exception as exc:  # noqa: BLE001 - judge failure should not abort
            self.last_error = f"{type(exc).__name__}: {exc}"
            # The longer plan is the safer fallback for multi-step tasks.
            choice = 1 if len(plan_a) >= len(plan_b) else 2
            self.last_choice = choice
            self.last_reason = "fallback: choose the longer plan"
            return plan_a if choice == 1 else plan_b
