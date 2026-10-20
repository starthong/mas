"""Generate a small tool-calling SFT set from templates.

This is deliberately synthetic: the goal is to teach output format and tool
selection, not world knowledge. Replace or extend the templates with real
queries before drawing conclusions about task quality.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.llm.prompts import build_system_prompt  # noqa: E402
from src.tools.builtin import build_default_registry  # noqa: E402

WEATHER_CITIES = [
    "北京", "上海", "广州", "深圳", "杭州",
    "成都", "武汉", "西安", "南京", "重庆",
    "苏州", "天津",
]
WEATHER_TEMPLATES = [
    "今天{city}天气怎么样？",
    "{city}现在天气如何？",
    "帮我查一下{city}的天气",
    "{city}今天出门需要带伞吗？",
    "我想知道{city}当前的气温",
    "查查{city}的天气情况",
    "{city}今天冷不冷？",
    "{city}今天热不热？",
    "{city}适合穿什么衣服？",
    "看看{city}今天的天气",
]

CALC_TEMPLATES = [
    "计算 {expr}",
    "帮我算一下 {expr}",
    "{expr} 等于多少？",
]

SEARCH_TOPICS = [
    "Transformer 架构", "LangGraph", "多智能体系统", "LoRA 微调",
    "ReAct 方法", "RAG 检索增强", "vLLM 推理", "QLoRA",
    "工具调用", "JSON Schema", "提示工程", "向量数据库",
    "BM25 检索", "Agent 记忆", "函数调用微调", "模型量化",
    "上下文窗口", "思维链", "监督微调", "评测指标",
]
SEARCH_TEMPLATES = [
    "搜索一下{topic}",
    "帮我查查{topic}",
    "我想了解{topic}",
    "介绍一下{topic}",
    "找一些关于{topic}的资料",
    "查一下{topic}是什么",
    "{topic}有什么资料",
    "帮我搜索{topic}",
]

FINAL_PROMPTS = [
    ("你好", "你好，有什么可以帮你？"),
    ("你是谁？", "我是这个多智能体系统里的执行助手。"),
    ("你能做什么？", "我可以拆解任务、调用工具并检查结果。"),
    ("谢谢", "不客气。"),
    ("再见", "再见，祝你顺利。"),
    ("早上好", "早上好。"),
    ("讲个笑话", "为什么程序员总分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。"),
    ("夸夸我", "你已经在做一件很有挑战性的事了，继续。"),
    ("今天适合学习吗？", "适合，先从最小的一步开始。"),
    ("你叫什么名字？", "叫我 mas 就行。"),
    ("帮我写一句鼓励的话", "慢一点没关系，别停。"),
    ("我有点焦虑", "先把任务拆小，完成一个算一个。"),
    ("晚安", "晚安，明天见。"),
    ("周末做什么？", "休息、复盘，再留一点时间给项目。"),
    ("推荐一本书", "《动手学深度学习》适合边做边学。"),
    ("谢谢你的帮助", "不用谢。"),
    ("你好呀", "你好。"),
    ("在吗", "在。"),
    ("我回来了", "欢迎回来。"),
    ("今天过得怎么样？", "还不错，正在等你的下一个任务。"),
]


def make_sample(system: str, user: str, name: str, arguments: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(
                {"name": name, "arguments": arguments},
                ensure_ascii=False,
            )},
        ]
    }


def random_expression(rng: random.Random) -> str:
    a = rng.randint(2, 99)
    b = rng.randint(2, 99)
    c = rng.randint(2, 20)
    kind = rng.choice(["mul_add", "paren", "div", "sub", "pow"])
    if kind == "mul_add":
        return f"{a}*{b}+{c}"
    if kind == "paren":
        return f"({a}+{b})*{c}"
    if kind == "div":
        return f"({a}*{c}+{b})/{c}"
    if kind == "sub":
        return f"{a}*{b}-{c}"
    return f"{a}+{b}*{c}"


def build_train(system: str, rng: random.Random) -> list[dict]:
    samples: list[dict] = []
    seen: set[str] = set()

    def add(user: str, name: str, args: dict) -> None:
        if user in seen:
            return
        seen.add(user)
        samples.append(make_sample(system, user, name, args))

    for city in WEATHER_CITIES:
        for template in WEATHER_TEMPLATES:
            add(template.format(city=city), "get_weather", {"city": city})

    for _ in range(150):
        expr = random_expression(rng)
        for template in CALC_TEMPLATES:
            add(template.format(expr=expr), "calculator", {"expression": expr})

    for topic in SEARCH_TOPICS:
        for template in SEARCH_TEMPLATES:
            add(template.format(topic=topic), "search", {"query": topic})

    for prompt, answer in FINAL_PROMPTS:
        add(prompt, "final_answer", {"answer": answer})

    rng.shuffle(samples)
    return samples


def build_test(system: str, rng: random.Random) -> list[dict]:
    samples: list[dict] = []
    seen: set[str] = set()

    def add(user: str, name: str, args: dict) -> None:
        if user in seen:
            return
        seen.add(user)
        samples.append(make_sample(system, user, name, args))

    # Held-out cities / topics / templates, so the test set is not a copy of train.
    for city in ["厦门", "青岛", "昆明"]:
        for template in ["{city}天气怎么样？", "查一下{city}的天气"]:
            add(template.format(city=city), "get_weather", {"city": city})

    for _ in range(20):
        expr = f"{rng.randint(100, 999)}/{rng.randint(2, 9)}"
        add(f"计算 {expr}", "calculator", {"expression": expr})

    for topic in ["知识图谱", "联邦学习", "模型蒸馏", "对比学习", "强化学习"]:
        for template in ["搜索一下{topic}", "介绍一下{topic}"]:
            add(template.format(topic=topic), "search", {"query": topic})

    for prompt, answer in [
        ("嗨", "嗨，我在。"),
        ("今天星期几？", "我这边没有日历工具，暂时查不了。"),
        ("给我讲个冷知识", "章鱼有三颗心脏。"),
        ("我该先做哪一步？", "先把最小的可验证步骤做完。"),
        ("谢谢，辛苦了", "不辛苦。"),
    ]:
        add(prompt, "final_answer", {"answer": answer})

    rng.shuffle(samples)
    return samples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    registry = build_default_registry()
    system = build_system_prompt(registry.describe())
    rng = random.Random(args.seed)

    train = build_train(system, rng)
    test = build_test(system, rng)

    write_jsonl(REPO_ROOT / "data" / "toolcall_sft.jsonl", train)
    write_jsonl(REPO_ROOT / "data" / "test" / "toolcall_test.jsonl", test)

    print(f"train: {len(train)} -> data/toolcall_sft.jsonl")
    print(f"test:  {len(test)} -> data/test/toolcall_test.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
