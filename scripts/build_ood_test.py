"""构造包含训练中未见工具的 OOD 测试集。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.llm.prompts import build_system_prompt  # noqa: E402
from src.tools.ood import build_ood_registry  # noqa: E402


def main() -> int:
    registry = build_ood_registry()
    system = build_system_prompt(registry.describe())

    cases: list[tuple[str, str, dict | None]] = []

    # 货币换算
    for user, amount, src, dst in [
        ("把 100 美元换成人民币", 100, "USD", "CNY"),
        ("500 元人民币能换多少美元", 500, "CNY", "USD"),
        ("把 50 欧元换成日元", 50, "EUR", "JPY"),
        ("1000 日元换成人民币", 1000, "JPY", "CNY"),
        ("帮我换 20 英镑成美元", 20, "GBP", "USD"),
        ("300 美元换成欧元", 300, "USD", "EUR"),
        ("100 港币换人民币", 100, "HKD", "CNY"),
        ("50 澳元换美元", 50, "AUD", "USD"),
    ]:
        cases.append((user, "currency_convert", {
            "amount": amount, "from_currency": src, "to_currency": dst
        }))

    # 日期加减
    for user, date, days in [
        ("2026-10-26 往后推 7 天", "2026-10-26", 7),
        ("2026-01-01 加 30 天", "2026-01-01", 30),
        ("2026-12-31 往后 1 天", "2026-12-31", 1),
        ("2026-03-01 减 10 天", "2026-03-01", -10),
        ("2026-05-20 加 100 天", "2026-05-20", 100),
        ("2026-10-01 往后 14 天", "2026-10-01", 14),
        ("2026-02-28 加 1 天", "2026-02-28", 1),
        ("2026-07-15 减 30 天", "2026-07-15", -30),
    ]:
        cases.append((user, "calendar_add", {"date": date, "days": days}))

    # 翻译
    for user, text, lang in [
        ("把“你好”翻译成英文", "你好", "English"),
        ("把“good morning”翻译成中文", "good morning", "Chinese"),
        ("把“谢谢”翻译成日文", "谢谢", "Japanese"),
        ("把“多智能体系统”翻译成英文", "多智能体系统", "English"),
        ("把“模型微调”翻译成法文", "模型微调", "French"),
        ("把“hello world”翻译成中文", "hello world", "Chinese"),
        ("把“再见”翻译成韩文", "再见", "Korean"),
        ("把“工具调用”翻译成英文", "工具调用", "English"),
    ]:
        cases.append((user, "translate", {"text": text, "target_language": lang}))

    # 股票查询
    for user, symbol in [
        ("查一下 AAPL 的股价", "AAPL"),
        ("腾讯控股现在股价多少", "0700.HK"),
        ("MSFT 股价", "MSFT"),
        ("特斯拉股票价格", "TSLA"),
        ("阿里巴巴美股价格", "BABA"),
        ("GOOGL 现在多少钱", "GOOGL"),
        ("亚马逊股价", "AMZN"),
        ("英伟达股票价格", "NVDA"),
    ]:
        cases.append((user, "get_stock_price", {"symbol": symbol}))

    # 缺参数、歧义、拒绝和无需工具。
    for user, answer in [
        ("帮我换一下货币", "请提供金额、源货币和目标货币。"),
        ("查一下苹果", "你是指苹果公司股票还是水果？请提供股票代码。"),
        ("帮我订一张去北京的机票", "当前没有订票工具，无法完成。"),
        ("把这句话翻译一下", "请提供要翻译的文本和目标语言。"),
        ("今天股价怎么样", "请提供具体的股票代码。"),
        ("帮我算一下", "请提供要计算的表达式。"),
        ("你好", "你好，有什么可以帮你？"),
        ("你能做什么？", "我可以做货币换算、日期计算、翻译和股票查询。"),
    ]:
        cases.append((user, "final_answer", {"answer": answer}))

    rows = []
    seen = set()
    for user, tool, args in cases:
        if user in seen:
            continue
        seen.add(user)
        rows.append(
            {
                "category": "ood",
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

    out = REPO_ROOT / "data" / "test" / "toolcall_ood.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"ood test: {len(rows)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
