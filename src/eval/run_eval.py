from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from src.eval.metrics import (
    classify_format_error,
    error_counts,
    exact_match,
    tool_name_match,
    validate_call,
    wilson_ci,
)
from src.llm.factory import build_client
from src.llm.parsing import parse_json_strict
from src.tools.builtin import build_default_registry
from src.tools.ood import build_ood_registry


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate tool-call JSON format.")
    parser.add_argument("--data", default="data/test/toolcall_test.jsonl")
    parser.add_argument("--backend", choices=["openai", "hf"], default=os.getenv("LLM_BACKEND", "openai"))
    parser.add_argument("--model", default="toolcall", help="OpenAI model name")
    parser.add_argument("--registry", choices=["default", "ood"], default="default")
    parser.add_argument("--base-model", default=os.getenv("HF_BASE_MODEL"))
    parser.add_argument("--adapter", default=os.getenv("HF_ADAPTER"))
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=os.getenv("HF_LOAD_IN_4BIT", "1") == "1",
    )
    parser.add_argument("--out", default="results/toolcall_eval.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    data_path = Path(args.data)
    rows = read_jsonl(data_path)
    if args.limit:
        rows = rows[: args.limit]

    registry = (
        build_ood_registry()
        if args.registry == "ood"
        else build_default_registry()
    )
    tools = registry.schemas()

    if args.backend == "hf":
        if not args.base_model:
            raise SystemExit("--base-model is required with --backend hf")
        client = build_client(
            "hf",
            base_model=args.base_model,
            adapter=args.adapter,
            load_in_4bit=args.load_in_4bit,
        )
        model_label = args.adapter or args.base_model
    else:
        client = build_client("openai")
        model_label = args.model

    records: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=f"eval {model_label}"):
        messages = row["messages"][:-1]
        ref_raw = row["messages"][-1]["content"]
        ref = json.loads(ref_raw) if isinstance(ref_raw, str) else ref_raw

        raw = ""
        error: str | None = None
        try:
            raw = client.chat(
                messages,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - record endpoint failures too
            error = f"{type(exc).__name__}: {exc}"

        parse_ok = False
        pred = None
        try:
            pred = parse_json_strict(raw)
            parse_ok = True
        except (json.JSONDecodeError, ValueError):
            pass

        records.append(
            {
                "raw": raw,
                "pred": pred,
                "ref": ref,
                "error": error,
                "json_parse_ok": parse_ok,
                "schema_ok": validate_call(pred, tools),
                "name_ok": tool_name_match(pred, ref),
                "exact": exact_match(pred, ref),
                "format_error": classify_format_error(raw),
            }
        )

    n = len(records)
    if n == 0:
        raise SystemExit(f"no records in {data_path}")

    schema_ok_count = sum(1 for item in records if item["schema_ok"])
    summary = {
        "model": model_label,
        "backend": args.backend,
        "registry": args.registry,
        "data": str(data_path),
        "n": n,
        "json_parse_rate": sum(1 for item in records if item["json_parse_ok"]) / n,
        "schema_rate": schema_ok_count / n,
        "schema_rate_ci95": wilson_ci(schema_ok_count, n),
        "name_accuracy": sum(1 for item in records if item["name_ok"]) / n,
        "exact_match_rate": sum(1 for item in records if item["exact"]) / n,
        "format_error_counts": error_counts(records),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"summary": summary, "records": records},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
