from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.llm.parsing import parse_json_loose
from src.llm.prompts import REVIEWER_SYSTEM

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class Reviewer:
    """Check whether the executor's results satisfy the task."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def review(
        self,
        task: str,
        plan: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        user = (
            f"Original task:\n{task}\n\n"
            f"Plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"Executor results:\n{json.dumps(results, ensure_ascii=False, indent=2)}\n\n"
            "Decide whether the task is complete. Return only the JSON object."
        )

        raw = ""
        try:
            raw = self.client.chat(
                [
                    {"role": "system", "content": REVIEWER_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            data = parse_json_loose(raw)
            issues = data.get("issues", [])
            if not isinstance(issues, list):
                issues = [str(issues)]
            return {
                "pass": bool(data.get("pass", False)),
                "score": float(data.get("score", 0.0)),
                "issues": [str(item) for item in issues],
                "suggestion": str(data.get("suggestion", "")),
                "raw": raw,
            }
        except Exception as exc:  # noqa: BLE001 - a broken review is a failed review
            return {
                "pass": False,
                "score": 0.0,
                "issues": [f"reviewer error: {type(exc).__name__}: {exc}"],
                "suggestion": "Re-plan and try again.",
                "raw": raw,
            }
