"""从 run_modes 输出生成按模式和温度的汇总表。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import wilson_ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    args = parser.parse_args()

    print("file\tgroup\tn\tsuccess\t95% CI\tlatency_s")
    for name in args.input:
        data = json.loads(Path(name).read_text(encoding="utf-8"))
        for group, row in data.get("summary", {}).get("groups", {}).items():
            n = int(row.get("n", 0))
            success = float(row.get("success_rate", 0.0))
            k = round(success * n)
            low, high = wilson_ci(k, n)
            latency = float(row.get("avg_latency_ms", 0.0)) / 1000
            print(f"{name}\t{group}\t{n}\t{success:.4f}\t[{low:.3f}, {high:.3f}]\t{latency:.2f}")


if __name__ == "__main__":
    main()
