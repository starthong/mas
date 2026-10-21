"""Compare two run_eval.py outputs and show the cases where they differ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["summary"], data["records"]


def rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for row in records if row.get(key)) / len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft", required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()

    ft_summary, ft = load_records(Path(args.ft))
    base_summary, base = load_records(Path(args.base))

    print(f"n: ft={len(ft)} base={len(base)}")
    print(f"model: ft={ft_summary.get('model')} base={base_summary.get('model')}")
    print()
    print(f"{'metric':<20} {'base':>8} {'ft':>8} {'delta':>8}")
    for key in ("json_parse_ok", "schema_ok", "name_ok", "exact"):
        b = rate(base, key)
        f = rate(ft, key)
        print(f"{key:<20} {b:>8.4f} {f:>8.4f} {f - b:>+8.4f}")

    limit = min(len(ft), len(base))
    print("\ndifferences:")
    shown = 0
    for index in range(limit):
        f = ft[index]
        b = base[index]
        if f.get("exact") == b.get("exact") and f.get("name_ok") == b.get("name_ok"):
            continue
        shown += 1
        print(f"\n[{index}]")
        print("ref :", json.dumps(f.get("ref"), ensure_ascii=False))
        print("base:", json.dumps(b.get("pred"), ensure_ascii=False))
        print("ft  :", json.dumps(f.get("pred"), ensure_ascii=False))
        print("base_error:", b.get("format_error"), "ft_error:", f.get("format_error"))

    if shown == 0:
        print("(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
