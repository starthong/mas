"""生成 25 条不参与前面调参的系统任务，用于最终 holdout。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "system_tasks_holdout.jsonl"
rows: list[dict] = []


def single(task_id: str, category: str, task: str, tool: str, arguments: dict | None) -> None:
    rows.append(
        {
            "id": task_id,
            "category": category,
            "task": task,
            "expected_tool": tool,
            "expected_arguments": arguments,
        }
    )


def multi(task_id: str, task: str, calls: list[dict]) -> None:
    rows.append({"id": task_id, "category": "multi", "task": task, "expected_calls": calls})


for city, idx in [("石家庄", 1), ("太原", 2), ("长春", 3), ("哈尔滨", 4)]:
    single(f"hold_weather_{idx:02d}", "weather", f"{city}今天天气怎么样？", "get_weather", {"city": city})

for expr, idx in [("19*6+7", 1), ("(81-9)/8", 2), ("13**2-20", 3), ("56/7+11", 4)]:
    single(f"hold_calc_{idx:02d}", "calculator", f"计算 {expr}", "calculator", {"expression": expr})

for topic, idx in [("GraphRAG", 1), ("MoE", 2), ("DPO", 3), ("FlashAttention", 4)]:
    single(f"hold_search_{idx:02d}", "search", f"搜索一下{topic}", "search", {"query": topic})

for prompt, idx in [
    ("推荐一个学习路线", 1),
    ("解释一下过拟合", 2),
    ("你有哪些限制", 3),
]:
    single(f"hold_chat_{idx:02d}", "chat", prompt, "final_answer", None)

multi(
    "hold_multi_weather_compare",
    "查石家庄和太原天气，比较温度",
    [
        {"tool": "get_weather", "arguments": {"city": "石家庄"}},
        {"tool": "get_weather", "arguments": {"city": "太原"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "hold_multi_search_summary",
    "搜索 GraphRAG，然后用两句话总结",
    [
        {"tool": "search", "arguments": {"query": "GraphRAG"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "hold_multi_calc_explain",
    "计算 19*6+7，然后解释结果",
    [
        {"tool": "calculator", "arguments": {"expression": "19*6+7"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "hold_multi_weather_calc",
    "查长春天气，然后计算 13*2",
    [
        {"tool": "get_weather", "arguments": {"city": "长春"}},
        {"tool": "calculator", "arguments": {"expression": "13*2"}},
    ],
)
multi(
    "hold_multi_search_compare",
    "搜索 MoE 和 DPO，比较区别",
    [
        {"tool": "search", "arguments": {"query": "MoE"}},
        {"tool": "search", "arguments": {"query": "DPO"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "hold_multi_weather_advice",
    "查哈尔滨天气，然后建议穿什么",
    [
        {"tool": "get_weather", "arguments": {"city": "哈尔滨"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "hold_multi_search_weather",
    "搜索 FlashAttention，再查石家庄天气",
    [
        {"tool": "search", "arguments": {"query": "FlashAttention"}},
        {"tool": "get_weather", "arguments": {"city": "石家庄"}},
    ],
)
multi(
    "hold_multi_calc_explain2",
    "计算 56/7+11，然后解释结果",
    [
        {"tool": "calculator", "arguments": {"expression": "56/7+11"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "hold_multi_weather_compare2",
    "查太原和哈尔滨天气，比较温度",
    [
        {"tool": "get_weather", "arguments": {"city": "太原"}},
        {"tool": "get_weather", "arguments": {"city": "哈尔滨"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "hold_multi_search_calc",
    "搜索 DPO，再计算 81/9",
    [
        {"tool": "search", "arguments": {"query": "DPO"}},
        {"tool": "calculator", "arguments": {"expression": "81/9"}},
    ],
)

OUT.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    encoding="utf-8",
)
print(f"wrote {len(rows)} rows -> {OUT}")
