from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import (
    classify_format_error,
    error_counts,
    exact_match,
    tool_name_match,
    validate_call,
    wilson_ci,
)
from src.llm.parsing import parse_json_strict
from src.tools.builtin import build_default_registry
from src.tools.ood import build_ood_registry

# 只约束顶层 JSON 协议；具体工具参数仍由 schema 校验。
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["name", "arguments"],
    "additionalProperties": False,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM guided-JSON 评测")
    parser.add_argument("--data", required=True)
    parser.add_argument("--registry", choices=["default", "ood"], default="default")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = read_jsonl(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]

    registry = build_ood_registry() if args.registry == "ood" else build_default_registry()
    tools = registry.schemas()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            row["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]

    llm = LLM(
        model=args.base_model,
        tokenizer=args.base_model,
        dtype="half",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enable_lora=bool(args.adapter),
        max_lora_rank=16,
    )

    try:
        from vllm.sampling_params import StructuredOutputsParams

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA),
        )
    except Exception:
        try:
            from vllm.sampling_params import GuidedDecodingParams

            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                guided_decoding=GuidedDecodingParams(json=JSON_SCHEMA),
            )
        except Exception:
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                guided_json=JSON_SCHEMA,
            )

    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("toolcall", 1, args.adapter)

    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    elapsed_ms = (time.perf_counter() - started) * 1000

    records = []
    for row, output in zip(rows, outputs):
        raw = output.outputs[0].text
        ref_raw = row["messages"][-1]["content"]
        ref = json.loads(ref_raw) if isinstance(ref_raw, str) else ref_raw
        try:
            pred = parse_json_strict(raw)
            parse_ok = True
        except (json.JSONDecodeError, ValueError):
            pred = None
            parse_ok = False
        records.append(
            {
                "raw": raw,
                "pred": pred,
                "ref": ref,
                "json_parse_ok": parse_ok,
                "schema_ok": validate_call(pred, tools),
                "name_ok": tool_name_match(pred, ref),
                "exact": exact_match(pred, ref),
                "format_error": classify_format_error(raw),
                "latency_ms": elapsed_ms / max(1, len(rows)),
            }
        )

    n = len(records)
    if n == 0:
        raise SystemExit("no records")
    schema_ok_count = sum(item["schema_ok"] for item in records)
    summary = {
        "model": args.adapter or args.base_model,
        "backend": "vllm-guided-json",
        "registry": args.registry,
        "data": args.data,
        "n": n,
        "json_parse_rate": sum(item["json_parse_ok"] for item in records) / n,
        "schema_rate": schema_ok_count / n,
        "schema_rate_ci95": wilson_ci(schema_ok_count, n),
        "name_accuracy": sum(item["name_ok"] for item in records) / n,
        "exact_match_rate": sum(item["exact"] for item in records) / n,
        "format_error_counts": error_counts(records),
        "avg_latency_ms": elapsed_ms / n,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
