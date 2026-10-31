"""在原始 SFT 数据后追加更多模板覆盖。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_toolcall_dataset import (  # noqa: E402
    CALC_TEMPLATES,
    make_train_sample,
    random_expression,
)
from src.llm.prompts import build_system_prompt  # noqa: E402
from src.tools.builtin import build_default_registry  # noqa: E402

NEW_WEATHER_CITIES = [
    "无锡", "宁波", "大连", "珠海", "佛山", "东莞",
    "沈阳", "哈尔滨", "石家庄", "南昌", "贵阳", "南宁",
]
NEW_SEARCH_TOPICS = [
    "机器学习", "深度学习", "神经网络", "注意力机制", "预训练",
    "指令微调", "模型对齐", "幻觉", "检索增强", "向量检索",
    "重排序", "提示词", "智能体", "任务规划", "反思",
    "多智能体", "消息传递", "协作", "评测", "鲁棒性",
]
NEW_FINAL = [
    ("早上好", "早上好，今天想先做哪件事？"),
    ("晚安", "晚安，明天继续。"),
    ("在吗", "在，直接说任务就行。"),
    ("我回来了", "欢迎回来，接着上次的进度继续。"),
    ("周末做什么", "休息、复盘，再留一点时间给项目。"),
    ("推荐一部电影", "《模仿游戏》适合对 AI 和历史都感兴趣的人。"),
    ("夸夸我", "你已经在把复杂问题拆成可验证的小步了，这很关键。"),
    ("我有点累", "先休息十分钟，再回来做最小的一步。"),
    ("今天适合学习吗", "适合，从最小的一步开始。"),
    ("你叫什么名字", "叫我 mas 就行。"),
    ("你能做什么", "我可以查天气、做四则运算和检索本地资料。"),
    ("帮我写一句鼓励的话", "慢一点没关系，别停。"),
    ("谢谢你的帮助", "不用谢。"),
    ("再见", "再见，祝你顺利。"),
    ("讲个冷笑话", "章鱼有三颗心脏。"),
    ("我该先做哪一步", "先把最小的可验证步骤做完。"),
    ("今天星期几", "我这边没有日历工具，暂时查不了。"),
    ("给我讲个笑话", "为什么程序员总分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。"),
    ("你好呀", "你好。"),
    ("谢谢", "不客气。"),
]


def main() -> int:
    source = ROOT / "data" / "toolcall_sft.jsonl"
    target = ROOT / "data" / "toolcall_sft_v2.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {
        row["conversations"][0]["value"]
        for row in rows
        if row.get("conversations")
    }

    registry = build_default_registry()
    system = build_system_prompt(registry.describe())
    rng = random.Random(2026)

    def add(user: str, name: str, args: dict) -> None:
        if user in seen:
            return
        seen.add(user)
        rows.append(make_train_sample(system, user, name, args))

    for city in NEW_WEATHER_CITIES:
        for template in ["{city}今天天气怎么样？", "帮我查一下{city}的天气"]:
            add(template.format(city=city), "get_weather", {"city": city})

    for topic in NEW_SEARCH_TOPICS:
        for template in ["搜索一下{topic}", "帮我查查{topic}"]:
            add(template.format(topic=topic), "search", {"query": topic})

    for _ in range(100):
        expr = random_expression(rng)
        for template in CALC_TEMPLATES:
            add(template.format(expr=expr), "calculator", {"expression": expr})

    for prompt, answer in NEW_FINAL:
        add(prompt, "final_answer", {"answer": answer})

    rng.shuffle(rows)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train v2: {len(rows)} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
