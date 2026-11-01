from __future__ import annotations

import re

from typing import TYPE_CHECKING, Any

from src.llm.parsing import parse_json_loose
from src.llm.prompts import PLANNER_SYSTEM, PLANNER_SYSTEM_FEWSHOT

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class Planner:
    """把任务拆成有序子任务。"""

    _MULTI_CUES = ("先", "再", "然后", "并", "分别", "比较", "同时")

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        few_shot: bool = False,
        system_prompt: str | None = None,
        samples: int = 1,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.few_shot = few_shot
        self.samples = max(1, samples)
        self.system_prompt = system_prompt or (
            PLANNER_SYSTEM_FEWSHOT if few_shot else PLANNER_SYSTEM
        )
        self.last_error: str | None = None

    def _parse_plan(self, raw: str) -> list[dict[str, Any]]:
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

    def _call(self, user: str, temperature: float) -> list[dict[str, Any]]:
        raw = self.client.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=512,
        )
        return self._parse_plan(raw)

    def plan(self, task: str, critique: str = "") -> list[dict[str, Any]]:
        self.last_error = None
        user = f"Task:\n{task}\n"
        if critique:
            user += f"\nThe reviewer rejected the previous attempt:\n{critique}\n"
            user += "Produce a revised plan that fixes these issues.\n"

        multi_cue = any(cue in task for cue in self._MULTI_CUES)
        plans: list[list[dict[str, Any]]] = []

        try:
            if self.samples > 1 and multi_cue:
                # 多动作任务用多次采样，选子任务最多的一份；不读取 gold 计划。
                for _ in range(self.samples):
                    plans.append(self._call(user, max(0.7, self.temperature)))
                plan = max(plans, key=len)
            else:
                plan = self._call(user, self.temperature)
        except Exception as exc:
            # 单条坏计划不应中断整个实验。
            self.last_error = f"{type(exc).__name__}: {exc}"
            return [{"id": 1, "goal": task, "expected_output": ""}]

        # few-shot Planner 在明确的多动作任务上偶尔仍只给一个子任务；
        # 此时追加一次“必须拆开”的提示，不依赖 gold 工具序列。
        if self.few_shot and len(plan) == 1 and multi_cue:
            retry_user = (
                user
                + "\nThis task contains multiple actions. Return at least two "
                "ordered subtasks; do not merge two lookups into one step.\n"
            )
            try:
                retry_plan = self._call(retry_user, self.temperature)
            except Exception:
                return plan
            if len(retry_plan) > 1:
                return retry_plan

        # 单个算式不应被拆成中间步骤，否则 exact match 会把正确结果判错。
        if (
            not multi_cue
            and len(plan) > 1
            and re.search(r"\d\s*[+\-*/]\s*\d", task)
        ):
            return [{"id": 1, "goal": task, "expected_output": ""}]

        return plan
