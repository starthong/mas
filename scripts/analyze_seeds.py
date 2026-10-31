"""汇总 v1/v2 的 seed 结果，输出均值和标准差。"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/seeds")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()

    root = Path(args.root)
    seeds = [int(item) for item in args.seeds.split(",")]
    for dataset in ("hard", "ood"):
        print(f"\n== {dataset}")
        groups: dict[str, list[dict]] = {"v1": [], "v2": []}
        for version in ("v1", "v2"):
            for seed in seeds:
                path = root / f"{version}-seed{seed}-{dataset}.json"
                summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
                groups[version].append(summary)
                print(
                    f"{version} seed{seed}: schema={summary['schema_rate']:.3f} "
                    f"name={summary['name_accuracy']:.3f} "
                    f"exact={summary['exact_match_rate']:.3f} "
                    f"lat={summary.get('avg_latency_ms', 0):.0f}ms"
                )
        for metric in ("schema_rate", "name_accuracy", "exact_match_rate"):
            for version in ("v1", "v2"):
                values = [row[metric] for row in groups[version]]
                print(
                    f"{version} {metric}: mean={statistics.mean(values):.4f} "
                    f"sd={statistics.stdev(values):.4f} "
                    f"range={min(values):.3f}-{max(values):.3f}"
                )
            diff = statistics.mean(row[metric] for row in groups["v2"]) - statistics.mean(
                row[metric] for row in groups["v1"]
            )
            print(f"v2-v1 {metric}: {diff:+.4f}")


if __name__ == "__main__":
    main()
