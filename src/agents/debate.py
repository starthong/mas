from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.agents.planner import Planner
from src.llm.parsing import parse_json_loose
from src.llm.prompts import JUDGE_SYSTEM, PLANNER_CRITIC_SYSTEM

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class DebatePlanner:
    """两个规划器分别提出计划，由裁判选择一个。"""

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        few_shot: bool = False,
        samples: int = 1,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.planner_a = Planner(
            client, temperature, few_shot=few_shot, samples=samples
        )
        self.planner_b = Planner(
            client,
            temperature,
            system_prompt=PLANNER_CRITIC_SYSTEM,
            samples=samples,
        )
        self.last_error: str | None = None
        self.last_choice: int | None = None
        self.last_reason = ""


    def _prefer_longer(self, task, selected, plan_a, plan_b):
        """多动作任务中，裁判偶尔把两个查询合并；此时回退到更长计划。"""
        if any(cue in task for cue in Planner._MULTI_CUES) and len(selected) == 1:
            longer = plan_a if len(plan_a) >= len(plan_b) else plan_b
            if len(longer) > 1:
                self.last_choice = 1 if longer is plan_a else 2
                self.last_reason = (self.last_reason + "；多动作任务选择更长计划").strip("；")
                return longer
        return selected

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
            selected = plan_a if choice == 1 else plan_b
            return self._prefer_longer(task, selected, plan_a, plan_b)
        except Exception as exc:  # noqa: BLE001 - judge failure should not abort
            self.last_error = f"{type(exc).__name__}: {exc}"
            # 多步任务下，选择更长的计划作为兜底。
            choice = 1 if len(plan_a) >= len(plan_b) else 2
            self.last_choice = choice
            self.last_reason = "fallback: choose the longer plan"
            selected = plan_a if choice == 1 else plan_b
            return self._prefer_longer(task, selected, plan_a, plan_b)
