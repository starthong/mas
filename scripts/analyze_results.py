"""Compute normalized OOD exact match, Wilson intervals, and McNemar tests."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

LANG = {
    "en": "English", "english": "English", "英文": "English", "英语": "English",
    "zh": "Chinese", "chinese": "Chinese", "中文": "Chinese", "汉语": "Chinese",
    "ja": "Japanese", "japanese": "Japanese", "日文": "Japanese", "日语": "Japanese",
    "fr": "French", "french": "French", "法文": "French", "法语": "French",
    "ko": "Korean", "korean": "Korean", "韩文": "Korean", "韩语": "Korean",
}
STOCK = {
    "腾讯控股": "0700.HK", "腾讯": "0700.HK", "特斯拉": "TSLA",
    "苹果": "AAPL", "微软": "MSFT", "谷歌": "GOOGL",
    "亚马逊": "AMZN", "英伟达": "NVDA", "阿里巴巴": "BABA",
}


def norm_call(call: dict) -> dict:
    name = call.get("name")
    args = call.get("arguments")
    if not isinstance(args, dict):
        return call
    args = dict(args)
    if name == "translate" and "target_language" in args:
        key = str(args["target_language"]).strip().lower()
        args["target_language"] = LANG.get(key, args["target_language"])
    if name == "get_stock_price" and "symbol" in args:
        key = str(args["symbol"]).strip()
        args["symbol"] = STOCK.get(key, key).upper()
    return {"name": name, "arguments": args}


def normalized_exact(row: dict) -> bool:
    ref, pred = row["ref"], row["pred"]
    if not isinstance(ref, dict) or not isinstance(pred, dict):
        return False
    if ref.get("name") == "final_answer":
        return False
    return norm_call(ref) == norm_call(pred)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def main() -> int:
    base = Path(__file__).resolve().parents[1] / "results" / "kaggle-3b"
    for prefix in ("hard", "ood"):
        print(f"\n== {prefix}")
        loaded = {}
        for temp in ("0", "07"):
            for model in ("base", "ft"):
                data = json.loads((base / f"{prefix}-{model}-{temp}.json").read_text(encoding="utf-8"))
                loaded[(model, temp)] = data
                rows = data["records"]
                n = len(rows)
                schema = sum(bool(r["schema_ok"]) for r in rows)
                name = sum(bool(r["name_ok"]) for r in rows)
                exact = sum(bool(r["exact"]) for r in rows)
                norm_rows = [r for r in rows if r["ref"].get("name") != "final_answer"]
                norm = sum(normalized_exact(r) for r in norm_rows)
                lo, hi = wilson(schema, n)
                print(
                    f"{model} temp {temp}: schema={schema}/{n} ({schema/n:.3f}, "
                    f"CI {lo:.3f}-{hi:.3f}) name={name}/{n} exact={exact}/{n} "
                    f"norm_exact={norm}/{len(norm_rows)}"
                )
                print("  errors", Counter(r["format_error"] for r in rows if r["format_error"] != "ok"))
        for temp in ("0", "07"):
            a = loaded[("base", temp)]["records"]
            b = loaded[("ft", temp)]["records"]
            base_ok = [bool(r["schema_ok"]) for r in a]
            ft_ok = [bool(r["schema_ok"]) for r in b]
            b01 = sum(1 for x, y in zip(base_ok, ft_ok) if x and not y)
            c10 = sum(1 for x, y in zip(base_ok, ft_ok) if not x and y)
            print(f"temp {temp}: base-only={b01}, ft-only={c10}, McNemar p={mcnemar_exact(b01, c10):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
