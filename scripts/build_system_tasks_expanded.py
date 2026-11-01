"""在原始 31 条系统任务后追加 24 条同协议任务，扩大到 55 条。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data" / "system_tasks.jsonl"
target = ROOT / "data" / "system_tasks_expanded.jsonl"

rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
seen = {row["id"] for row in rows}
seen_text = {row["task"] for row in rows}


def add(row: dict) -> None:
    if row["id"] in seen or row["task"] in seen_text:
        raise ValueError(f"duplicate task: {row}")
    seen.add(row["id"])
    seen_text.add(row["task"])
    rows.append(row)


def single(task_id: str, category: str, task: str, tool: str, arguments: dict) -> None:
    add(
        {
            "id": task_id,
            "category": category,
            "task": task,
            "expected_tool": tool,
            "expected_arguments": arguments,
        }
    )


def multi(task_id: str, task: str, calls: list[dict]) -> None:
    add({"id": task_id, "category": "multi", "task": task, "expected_calls": calls})


for city, idx in [("深圳", 25), ("杭州", 26), ("成都", 27), ("武汉", 28)]:
    single(f"weather_{idx:02d}", "weather", f"{city}今天天气怎么样？", "get_weather", {"city": city})

for expr, idx in [
    ("12.5*4+3", 29),
    ("(99-17)/4", 30),
    ("7**3", 31),
    ("45/9+8", 32),
]:
    single(f"calc_{idx:02d}", "calculator", f"计算 {expr}", "calculator", {"expression": expr})

for topic, idx in [("RAG", 33), ("vLLM", 34), ("Agent 记忆", 35), ("BM25", 36)]:
    single(f"search_{idx:02d}", "search", f"搜索一下{topic}", "search", {"query": topic})

for idx, (prompt, answer) in enumerate(
    [
        ("你平时怎么工作？", "我会先拆解任务，再调用工具，最后检查结果。"),
        ("推荐一部电影", "《模仿游戏》适合对 AI 和历史感兴趣的人。"),
        ("现在几点？", "当前没有时钟工具，无法查询。"),
        ("我有点累", "先休息十分钟，再回来做最小的一步。"),
        ("辛苦你了", "不辛苦，继续。"),
    ],
    start=37,
):
    single(f"final_{idx:02d}", "chat", prompt, "final_answer", None)

multi(
    "multi_weather_advice_expanded",
    "先查上海天气，再根据温度建议穿什么",
    [
        {"tool": "get_weather", "arguments": {"city": "上海"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "multi_search_summary_expanded",
    "搜索 RAG，然后用两句话总结",
    [
        {"tool": "search", "arguments": {"query": "RAG"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "multi_calc_explain_expanded",
    "计算 45/9+8，然后解释结果",
    [
        {"tool": "calculator", "arguments": {"expression": "45/9+8"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "multi_weather_compare_expanded",
    "查广州和深圳天气，比较温度",
    [
        {"tool": "get_weather", "arguments": {"city": "广州"}},
        {"tool": "get_weather", "arguments": {"city": "深圳"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "multi_search_compare_expanded",
    "搜索 RAG 和 vLLM，比较区别",
    [
        {"tool": "search", "arguments": {"query": "RAG"}},
        {"tool": "search", "arguments": {"query": "vLLM"}},
        {"tool": "final_answer", "arguments": None},
    ],
)
multi(
    "multi_weather_calc_expanded",
    "查上海天气，然后计算 12.5*4",
    [
        {"tool": "get_weather", "arguments": {"city": "上海"}},
        {"tool": "calculator", "arguments": {"expression": "12.5*4"}},
    ],
)
multi(
    "multi_search_weather_expanded",
    "搜索 BM25，再查杭州天气",
    [
        {"tool": "search", "arguments": {"query": "BM25"}},
        {"tool": "get_weather", "arguments": {"city": "杭州"}},
    ],
)

with target.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"tasks: {len(rows)} -> {target}")
print(f"single: {sum(1 for row in rows if row['category'] != 'multi')}")
print(f"multi: {sum(1 for row in rows if row['category'] == 'multi')}")
