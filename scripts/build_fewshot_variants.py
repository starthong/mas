"""为 hard 和 OOD 测试集生成 few-shot 版本。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "test"

FEWSHOT = {
    "toolcall_hard": [
        ("计算 2+3", {"name": "calculator", "arguments": {"expression": "2+3"}}),
        ("查一下北京天气", {"name": "get_weather", "arguments": {"city": "北京"}}),
        ("你好", {"name": "final_answer", "arguments": {"answer": "你好，有什么可以帮你？"}}),
    ],
    "toolcall_ood": [
        ("把 100 美元换成人民币", {"name": "currency_convert", "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "CNY"}}),
        ("2026-10-26 往后推 7 天", {"name": "calendar_add", "arguments": {"date": "2026-10-26", "days": 7}}),
        ("把“你好”翻译成英文", {"name": "translate", "arguments": {"text": "你好", "target_language": "English"}}),
    ],
}


def build(name: str) -> int:
    source = TEST / f"{name}.jsonl"
    target = TEST / f"{name}_fewshot.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = FEWSHOT[name]
    out = []
    for row in rows:
        messages = list(row["messages"])
        system = messages[0]
        shot_messages = [system]
        for user, call in examples:
            shot_messages.append({"role": "user", "content": user})
            shot_messages.append({"role": "assistant", "content": json.dumps(call, ensure_ascii=False)})
        shot_messages.extend(messages[1:])
        out.append({**row, "messages": shot_messages})
    with target.open("w", encoding="utf-8") as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{target}: {len(out)}")
    return 0


if __name__ == "__main__":
    build("toolcall_hard")
    build("toolcall_ood")
