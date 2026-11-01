"""从 Planner 合成数据生成 Reviewer 正/负样本。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.prompts import REVIEWER_SYSTEM

SOURCE = ROOT / "data" / "planner_sft.jsonl"
TARGET = ROOT / "data" / "reviewer_sft.jsonl"
rows: list[dict] = []


def result_rows(plan: dict, valid: bool) -> list[dict]:
    results = []
    for index, item in enumerate(plan.get("subtasks", [])):
        text = "执行结果：已完成该子任务。" if valid else ""
        results.append(
            {
                "subtask_id": item.get("id", index + 1),
                "goal": item.get("goal", ""),
                "tool": "tool",
                "arguments": {},
                "result": text,
                "error": None if valid else "empty result",
            }
        )
    if not valid and results:
        results = results[:-1]
    return results


for line in SOURCE.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    human = next(item["value"] for item in row["conversations"] if item["from"] == "human")
    target = next(item["value"] for item in row["conversations"] if item["from"] == "gpt")
    task = human.split("Task:\n", 1)[-1].strip()
    plan = json.loads(target)

    for valid in (True, False):
        results = result_rows(plan, valid)
        user = (
            f"Original task:\n{task}\n\n"
            f"Plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"Executor results:\n{json.dumps(results, ensure_ascii=False, indent=2)}\n\n"
            "Decide whether the task is complete. Return only the JSON object."
        )
        answer = {
            "pass": valid,
            "score": 0.95 if valid else 0.2,
            "issues": [] if valid else ["执行结果缺失或不完整"],
            "suggestion": "" if valid else "补全缺失的子任务后重新审查。",
        }
        rows.append(
            {
                "system": REVIEWER_SYSTEM,
                "conversations": [
                    {"from": "human", "value": user},
                    {"from": "gpt", "value": json.dumps(answer, ensure_ascii=False)},
                ],
            }
        )

TARGET.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    encoding="utf-8",
)
print(f"wrote {len(rows)} rows -> {TARGET}")
