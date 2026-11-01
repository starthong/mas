"""生成较大规模的 Planner SFT 数据；任务模板避开 55 条和 holdout。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.prompts import PLANNER_SYSTEM_FEWSHOT

OUT = ROOT / "data" / "planner_sft.jsonl"
rows: list[dict] = []
seen_tasks: set[str] = set()


def plan(*subtasks: tuple[str, str]) -> dict:
    return {
        "subtasks": [
            {"id": index, "goal": goal, "expected_output": output}
            for index, (goal, output) in enumerate(subtasks, start=1)
        ]
    }


def add(task: str, subtasks: tuple[tuple[str, str], ...]) -> None:
    if task in seen_tasks:
        return
    seen_tasks.add(task)
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


# 避开 55 条和 holdout 使用的主城市。
cities = [
    "南京", "苏州", "天津", "重庆", "长沙", "厦门", "青岛", "昆明",
    "济南", "合肥", "福州", "南昌", "贵阳", "南宁", "海口", "兰州",
    "银川", "乌鲁木齐", "郑州", "西安", "沈阳", "大连", "无锡", "宁波",
    "佛山", "东莞", "珠海", "三亚",
]
weather_templates = [
    "{city}今天天气怎么样？",
    "帮我查查{city}的天气",
    "{city}现在多少度？",
    "看看{city}的天气",
]
for city in cities:
    for template in weather_templates:
        add(template.format(city=city), ((f"查询{city}的天气", "天气信息"),))

random.seed(20261101)
exprs: list[str] = []
for _ in range(80):
    a = random.randint(2, 40)
    b = random.randint(2, 12)
    op = random.choice(["+", "-", "*", "/"])
    if op == "/":
        a = a * b
        expr = f"{a}/{b}"
    else:
        expr = f"{a}{op}{b}"
    if expr not in exprs:
        exprs.append(expr)
for expr in exprs:
    add(f"计算 {expr}", ((f"计算表达式 {expr}", "数值结果"),))
    add(f"{expr} 的结果是多少？", ((f"计算表达式 {expr}", "数值结果"),))

topics = [
    "向量数据库", "混合检索", "重排序", "语义缓存", "提示压缩",
    "工具学习", "函数调用", "多模态检索", "长上下文", "推理缓存",
    "模型量化", "知识蒸馏", "稀疏激活", "检索增强", "文档切分",
    "嵌入模型", "对比学习", "指令微调", "奖励模型", "偏好优化",
    "智能体规划", "任务分解", "工具路由", "记忆机制", "反思机制",
    "多智能体", "图数据库", "倒排索引", "近似最近邻", "查询扩展",
]
for topic in topics:
    add(f"搜索一下{topic}", ((f"搜索{topic}的相关资料", "检索结果"),))
    add(f"帮我查查{topic}的资料", ((f"搜索{topic}的相关资料", "检索结果"),))

chats = [
    "给我一句鼓励", "解释一下过拟合", "推荐一部纪录片", "你更擅长什么",
    "现在适合做什么", "用一句话解释 RAG", "给我一个学习建议",
    "推荐一个开源项目", "你有哪些限制", "怎么安排一天的学习",
    "什么是梯度消失", "为什么要做数据清洗", "怎么评估检索质量",
    "什么是指令遵循", "如何避免过拟合", "解释一下注意力机制",
    "推荐一本技术书", "我应该先学 Python 还是数学", "怎么读论文",
    "如何做消融实验",
]
for prompt in chats:
    add(prompt, (("直接回答用户的问题", "回答"),))

for index in range(len(cities) - 1):
    left, right = cities[index], cities[index + 1]
    add(
        f"先查{left}天气，再查{right}天气，比较温度",
        (
            (f"查询{left}的天气", "天气信息"),
            (f"查询{right}的天气", "天气信息"),
            (f"比较{left}和{right}的温度", "比较结果"),
        ),
    )

for index in range(0, min(len(topics), len(exprs)), 2):
    topic, expr = topics[index], exprs[index]
    add(
        f"搜索{topic}，然后计算 {expr}",
        (
            (f"搜索{topic}的相关资料", "检索结果"),
            (f"计算表达式 {expr}", "数值结果"),
        ),
    )

for index in range(len(cities) - 2):
    city, topic = cities[index], topics[index % len(topics)]
    add(
        f"先搜索{topic}，再查{city}天气",
        (
            (f"搜索{topic}的相关资料", "检索结果"),
            (f"查询{city}的天气", "天气信息"),
        ),
    )

for index in range(0, min(len(cities), len(exprs)), 2):
    city, expr = cities[index], exprs[index]
    add(
        f"查{city}天气，然后计算 {expr}",
        (
            (f"查询{city}的天气", "天气信息"),
            (f"计算表达式 {expr}", "数值结果"),
        ),
    )

for index in range(0, min(len(topics) - 1, 20)):
    left, right = topics[index], topics[index + 1]
    add(
        f"搜索 {left} 和 {right}，比较区别",
        (
            (f"搜索{left}的相关资料", "检索结果"),
            (f"搜索{right}的相关资料", "检索结果"),
            (f"比较{left}和{right}的区别", "比较结果"),
        ),
    )

for index in range(0, min(len(chats), len(exprs)), 2):
    chat, expr = chats[index], exprs[index]
    add(
        f"计算 {expr}，然后{chat}",
        (
            (f"计算表达式 {expr}", "数值结果"),
            (f"{chat}", "最终回答"),
        ),
    )

# 请求前缀复述，进一步扩大规模；不改变任务语义。
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
