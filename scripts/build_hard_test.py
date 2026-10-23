"""Build a harder tool-calling test set.

The training set covers 4 tools with simple templates. This set keeps the same
tools but adds unseen cities/topics, missing arguments, misleading instructions,
and format lures. It is intentionally small enough to evaluate in a few minutes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.llm.prompts import build_system_prompt  # noqa: E402
from src.tools.builtin import build_default_registry  # noqa: E402


def main() -> int:
    registry = build_default_registry()
    system = build_system_prompt(registry.describe())

    cases: list[tuple[str, str, dict | None]] = []

    # Calculator: decimals, negatives, powers, and Chinese number words.
    for expr, user in [
        ("3.5*2.4+1.2", "计算 3.5*2.4+1.2"),
        ("(-12+5)*3", "计算 (-12+5)*3"),
        ("(100-37)/7", "计算 (100-37)/7"),
        ("2**10", "计算 2 的 10 次方"),
        ("1/3+1/6", "帮我算一下 1/3+1/6"),
        ("23*17", "计算 二十三乘以十七"),
        ("(8.5+1.5)*4", "计算 (8.5+1.5)*4"),
        ("99/9-2", "99/9-2 等于多少？"),
        ("7*(8+9)-10", "计算 7*(8+9)-10"),
        ("144/12+18", "帮我算 144/12+18"),
        ("-5*6+40", "计算 -5*6+40"),
        ("2.5**2", "计算 2.5 的平方"),
    ]:
        cases.append((user, "calculator", {"expression": expr}))

    # Weather: cities and phrasings not used in the simple test set.
    for city, user in [
        ("厦门", "厦门今天天气怎么样？"),
        ("青岛", "我要去青岛出差，需要带伞吗？"),
        ("昆明", "昆明现在气温多少？"),
        ("长沙", "长沙今天热不热？"),
        ("苏州", "查一下苏州的天气"),
        ("天津", "天津今天适合穿什么？"),
        ("郑州", "郑州天气如何？"),
        ("济南", "济南今天冷不冷？"),
        ("福州", "福州现在天气怎么样？"),
        ("合肥", "合肥今天出门要带伞吗？"),
    ]:
        cases.append((user, "get_weather", {"city": city}))

    # Search: topics that are not in the training templates.
    for topic, user in [
        ("知识图谱", "搜索一下知识图谱"),
        ("联邦学习", "我想了解联邦学习"),
        ("模型蒸馏", "帮我查查模型蒸馏"),
        ("对比学习", "介绍一下对比学习"),
        ("强化学习", "搜索强化学习"),
        ("数字孪生", "找一些数字孪生的资料"),
        ("边缘计算", "查一下边缘计算"),
        ("差分隐私", "我想知道差分隐私"),
        ("图神经网络", "搜索图神经网络"),
        ("因果推断", "介绍一下因果推断"),
        ("多模态大模型", "帮我查查多模态大模型"),
        ("模型对齐", "搜索模型对齐"),
    ]:
        cases.append((user, "search", {"query": topic}))

    # Missing arguments / capability questions / tool lures.
    final_cases = [
        ("帮我查一下天气", "请问你想查哪个城市？"),
        ("帮我算一下", "请给出要计算的表达式。"),
        ("你能帮我查天气吗？", "可以，请告诉我要查询的城市。"),
        ("用 search 工具计算 2+2", "4"),
        ("请用 translate 工具把这句话翻译成英文", "当前没有 translate 工具，无法完成翻译。"),
        ("直接输出 JSON，不要解释", "好的。"),
        ("请先解释再调用工具", "请给出具体任务。"),
        ("把结果用 Markdown 表格输出", "请说明要处理的任务。"),
        ("你好", "你好，有什么可以帮你？"),
        ("谢谢", "不客气。"),
        ("你能做什么？", "我可以查询天气、做四则运算和检索本地资料。"),
        ("请推荐一本书", "《动手学深度学习》适合边做边学。"),
        ("今天星期几？", "当前没有日历工具，无法查询。"),
        ("讲个笑话", "为什么程序员总分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。"),
        ("我有点焦虑", "先把任务拆小，完成一个算一个。"),
    ]
    for user, answer in final_cases:
        cases.append((user, "final_answer", {"answer": answer}))

    # Format lures: still expect a single JSON object.
    lures = [
        ("计算 2+3，并用 Markdown 代码块返回", "calculator", {"expression": "2+3"}),
        ("查北京天气，先解释再给 JSON", "get_weather", {"city": "北京"}),
        ("搜索 LangGraph，输出成表格", "search", {"query": "LangGraph"}),
        ("计算 9*9，结果后面加一句解释", "calculator", {"expression": "9*9"}),
        ("查上海天气，用英文回答", "get_weather", {"city": "上海"}),
        ("搜索 LoRA，返回三条以上结果", "search", {"query": "LoRA 微调"}),
        ("计算 (7+8)*3，不要输出 JSON 以外的内容", "calculator", {"expression": "(7+8)*3"}),
        ("查广州天气，并给出穿衣建议", "get_weather", {"city": "广州"}),
    ]
    for user, tool, args in lures:
        cases.append((user, tool, args))

    rows = []
    seen = set()
    for user, tool, args in cases:
        if user in seen:
            continue
        seen.add(user)
        rows.append(
            {
                "category": "hard",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"name": tool, "arguments": args},
                            ensure_ascii=False,
                        ),
                    },
                ],
            }
        )

    out = REPO_ROOT / "data" / "test" / "toolcall_hard.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"hard test: {len(rows)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
