"""生成 Planner 专用 SFT 数据；任务模板与评测集不重合。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.prompts import PLANNER_SYSTEM_FEWSHOT

OUT = ROOT / "data" / "planner_sft.jsonl"
rows: list[dict] = []


def plan(*subtasks: tuple[str, str]) -> dict:
    return {
        "subtasks": [
            {"id": index, "goal": goal, "expected_output": output}
            for index, (goal, output) in enumerate(subtasks, start=1)
        ]
    }


def add(task: str, subtasks: tuple[tuple[str, str], ...]) -> None:
    rows.append(
        {
            "system": PLANNER_SYSTEM_FEWSHOT,
            "conversations": [
                {"from": "human", "value": f"Task:\n{task}\n"},
                {
                    "from": "gpt",
                    "value": json.dumps(plan(*subtasks), ensure_ascii=False),
                },
            ],
        }
    )


weather_cities = ["南京", "苏州", "天津", "重庆", "长沙", "厦门", "青岛", "昆明"]
for city in weather_cities:
    add(
        f"{city}今天天气怎么样？",
        ((f"查询{city}今天的天气", "天气信息"),),
    )
    add(
        f"帮我看看{city}现在多少度",
        ((f"查询{city}当前的温度", "温度信息"),),
    )

calc_exprs = [
    "18*7+6",
    "(64-16)/8",
    "9**2-11",
    "72/9+14",
    "31*4-25",
    "125/5+7",
]
for expr in calc_exprs:
    add(
        f"计算 {expr}",
        ((f"计算表达式 {expr}", "数值结果"),),
    )

search_topics = ["向量数据库", "混合检索", "BM25", "LangGraph", "ReAct", "QLoRA", "重排序"]
for topic in search_topics:
    add(
        f"搜索一下{topic}",
        ((f"搜索{topic}的相关资料", "检索结果"),),
    )

chat_prompts = [
    "给我一句鼓励",
    "解释一下什么是过拟合",
    "推荐一部纪录片",
    "现在适合做什么",
    "你更擅长什么",
]
for prompt in chat_prompts:
    add(
        prompt,
        (("直接回答用户的问题", "回答"),),
    )

for index, city in enumerate(weather_cities[:4]):
    other = weather_cities[(index + 1) % len(weather_cities)]
    add(
        f"先查{city}天气，再查{other}天气，比较温度",
        (
            (f"查询{city}的天气", "天气信息"),
            (f"查询{other}的天气", "天气信息"),
            (f"比较{city}和{other}的温度", "比较结果"),
        ),
    )

for index, topic in enumerate(search_topics[:4]):
    expr = calc_exprs[index]
    add(
        f"搜索{topic}，然后计算 {expr}",
        (
            (f"搜索{topic}的相关资料", "检索结果"),
            (f"计算表达式 {expr}", "数值结果"),
        ),
    )

for index, city in enumerate(weather_cities[2:6]):
    topic = search_topics[(index + 1) % len(search_topics)]
    add(
        f"先搜索{topic}，再查{city}天气",
        (
            (f"搜索{topic}的相关资料", "检索结果"),
            (f"查询{city}的天气", "天气信息"),
        ),
    )

for prompt, expr in zip(chat_prompts[:3], calc_exprs[:3]):
    add(
        f"计算 {expr}，然后{prompt}",
        (
            (f"计算表达式 {expr}", "数值结果"),
            (f"{prompt}", "最终回答"),
        ),
    )

# 额外模板：增加城市、算式和主题覆盖，仍避开 55 条评测集的主模板。
extra_cities = ["济南", "合肥", "福州", "南昌", "贵阳", "南宁", "海口", "兰州", "银川", "乌鲁木齐"]
for city in extra_cities:
    add(f"{city}天气如何？", ((f"查询{city}的天气", "天气信息"),))
    add(f"查一下{city}现在的温度", ((f"查询{city}当前的温度", "温度信息"),))

extra_exprs = [
    "36*5+12",
    "(120-24)/8",
    "11**2+19",
    "84/7-3",
    "27*3+18",
    "144/12+9",
    "15*6-21",
    "96/8+17",
]
for expr in extra_exprs:
    add(f"帮我计算 {expr}", ((f"计算表达式 {expr}", "数值结果"),))

extra_topics = ["稠密检索", "稀疏检索", "语义缓存", "提示压缩", "工具学习", "函数调用", "规划器", "审查器"]
for topic in extra_topics:
    add(f"查一下{topic}的资料", ((f"搜索{topic}的相关资料", "检索结果"),))

extra_chats = [
    "用一句话解释什么是 RAG",
    "给我一个学习建议",
    "推荐一个开源项目",
    "现在适合休息吗",
    "你有哪些限制",
]
for prompt in extra_chats:
    add(prompt, (("直接回答用户的问题", "回答"),))

for left, right in zip(extra_cities[:5], extra_cities[5:]):
    add(
        f"查{left}和{right}的天气，比较温度",
        (
            (f"查询{left}的天气", "天气信息"),
            (f"查询{right}的天气", "天气信息"),
            (f"比较{left}和{right}的温度", "比较结果"),
        ),
    )

for topic, expr in zip(extra_topics[:5], extra_exprs[:5]):
    add(
        f"搜索{topic}，再计算 {expr}",
        (
            (f"搜索{topic}的相关资料", "检索结果"),
            (f"计算表达式 {expr}", "数值结果"),
        ),
    )

for city, topic in zip(extra_cities[:5], extra_topics[3:8]):
    add(
        f"先查{topic}，再查{city}天气",
        (
            (f"搜索{topic}的相关资料", "检索结果"),
            (f"查询{city}的天气", "天气信息"),
        ),
    )

for prompt, expr in zip(extra_chats[:5], extra_exprs[3:8]):
    add(
        f"计算 {expr}，然后{prompt}",
        (
            (f"计算表达式 {expr}", "数值结果"),
            (f"{prompt}", "最终回答"),
        ),
    )

# 轻量复述增强：同一组合成计划换两种请求前缀，不引入评测任务。
base_rows = list(rows)
for prefix in ("请帮我", "麻烦你"):
    for row in base_rows:
        cloned = json.loads(json.dumps(row, ensure_ascii=False))
        human = cloned["conversations"][0]["value"]
        cloned["conversations"][0]["value"] = human.replace(
            "Task:\n", f"Task:\n{prefix}", 1
        )
        rows.append(cloned)

OUT.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    encoding="utf-8",
)
print(f"wrote {len(rows)} rows -> {OUT}")
