"""Run single-agent vs sequential multi-agent experiments on system tasks."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.agents.executor import Executor
from src.eval.metrics import validate_call
from src.graph.workflow import build_graph
from src.llm.factory import build_client
from src.llm.parsing import parse_json_strict
from src.tools.builtin import build_default_registry


def load_tasks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def result_matches(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    if result.get("error"):
        return False
    if result.get("tool") != expected.get("tool"):
        return False
    arguments = expected.get("arguments")
    if arguments is None:
        return True
    return result.get("arguments") == arguments


def task_success(results: list[dict[str, Any]], task: dict[str, Any]) -> bool:
    expected_calls = task.get("expected_calls")
    if expected_calls is None:
        expected_calls = [
            {
                "tool": task.get("expected_tool"),
                "arguments": task.get("expected_arguments"),
            }
        ]
    return all(
        any(result_matches(result, expected) for result in results)
        for expected in expected_calls
    )


def run_single(
    client: Any,
    registry: Any,
    task: dict[str, Any],
    temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    executor = Executor(client, registry, temperature=temperature)
    result = executor.run(
        task["task"],
        {"id": 1, "goal": task["task"]},
    )
    return [result], {"rounds": 1, "review": {}, "trace": ["single"]}


def run_sequential(
    client: Any,
    registry: Any,
    task: dict[str, Any],
    temperature: float,
    planner_client: Any = None,
    reviewer_client: Any = None,
    enable_reviewer: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = build_graph(
        client,
        registry,
        max_rounds=2,
        temperature=temperature,
        planner_client=planner_client,
        reviewer_client=reviewer_client,
        enable_reviewer=enable_reviewer,
    )
    state = graph.invoke(
        {
            "task": task["task"],
            "plan": [],
            "results": [],
            "review": {},
            "rounds": 0,
            "trace": [],
        }
    )
    return state.get("results", []), state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/system_tasks.jsonl")
    parser.add_argument("--modes", default="single,sequential")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperatures", default="0")
    parser.add_argument("--backend", choices=["openai", "hf"], default="hf")
    parser.add_argument("--model", default="toolcall")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument(
        "--reviewer",
        choices=["lora", "ft", "base", "none"],
        default="lora",
        help="lora/ft use the fine-tuned model; ft is kept as an alias",
    )
    parser.add_argument(
        "--planner",
        choices=["lora", "ft", "base"],
        default="lora",
        help="lora/ft use the fine-tuned model; ft is kept as an alias",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/modes.json")
    args = parser.parse_args()

    planner_mode = "lora" if args.planner in ("lora", "ft") else "base"
    reviewer_mode = "lora" if args.reviewer in ("lora", "ft") else args.reviewer

    tasks = load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    temperatures = [float(item) for item in args.temperatures.split(",") if item.strip()]

    registry = build_default_registry()
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
        client = build_client("openai", default_model=args.model)
        model_label = args.model

    base_client = None
    if planner_mode == "base" or reviewer_mode == "base":
        if args.backend != "hf":
            raise SystemExit("base planner/reviewer is only supported with --backend hf")
        base_client = build_client(
            "hf",
            base_model=args.base_model,
            adapter=None,
            load_in_4bit=args.load_in_4bit,
        )

    planner_client = base_client if planner_mode == "base" else None
    reviewer_client = base_client if reviewer_mode == "base" else None
    enable_reviewer = reviewer_mode != "none"

    records: list[dict[str, Any]] = []
    for mode in modes:
        for temperature in temperatures:
            for task in tasks:
                for run_index in range(1, args.runs + 1):
                    started = time.perf_counter()
                    run_error: str | None = None
                    try:
                        if mode == "single":
                            results, state = run_single(
                                client, registry, task, temperature
                            )
                        elif mode == "sequential":
                            results, state = run_sequential(
                                client,
                                registry,
                                task,
                                temperature,
                                planner_client=planner_client,
                                reviewer_client=reviewer_client,
                                enable_reviewer=enable_reviewer,
                            )
                        else:
                            raise SystemExit(f"unknown mode: {mode}")
                    except Exception as exc:  # noqa: BLE001 - one task must not abort the run
                        results = []
                        state = {}
                        run_error = f"{type(exc).__name__}: {exc}"
                    latency_ms = (time.perf_counter() - started) * 1000

                    parsed: list[dict[str, Any] | None] = []
                    for result in results:
                        raw = result.get("raw", "")
                        try:
                            parsed.append(parse_json_strict(raw))
                        except (json.JSONDecodeError, ValueError):
                            parsed.append(None)

                    json_ok = bool(parsed) and all(item is not None for item in parsed)
                    schema_ok = bool(parsed) and all(
                        validate_call(item, tools) for item in parsed
                    )
                    success = task_success(results, task)
                    review = state.get("review") or {}

                    records.append(
                        {
                            "mode": mode,
                            "planner": planner_mode,
                            "reviewer": reviewer_mode,
                            "temperature": temperature,
                            "run": run_index,
                            "task_id": task.get("id"),
                            "category": task.get("category"),
                            "task": task.get("task"),
                            "expected_tool": task.get("expected_tool"),
                            "expected_calls": task.get("expected_calls"),
                            "success": success,
                            "json_parse_ok": json_ok,
                            "schema_ok": schema_ok,
                            "review_pass": review.get("pass"),
                            "rounds": state.get("rounds"),
                            "num_results": len(results),
                            "latency_ms": latency_ms,
                            "results": results,
                            "trace": state.get("trace", []),
                            "review": review,
                            "run_error": run_error,
                        }
                    )

    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["mode"], record["temperature"])].append(record)

    summary: dict[str, Any] = {
        "model": model_label,
        "backend": args.backend,
        "planner": planner_mode,
        "reviewer": reviewer_mode,
        "tasks": len(tasks),
        "runs": args.runs,
        "groups": {},
    }
    for (mode, temperature), rows in groups.items():
        n = len(rows)
        review_rows = [row for row in rows if row.get("review_pass") is not None]
        summary["groups"][f"{mode}@{temperature}"] = {
            "n": n,
            "success_rate": sum(1 for row in rows if row["success"]) / n,
            "json_parse_rate": sum(1 for row in rows if row["json_parse_ok"]) / n,
            "schema_rate": sum(1 for row in rows if row["schema_ok"]) / n,
            "review_pass_rate": (
                sum(1 for row in review_rows if row["review_pass"]) / len(review_rows)
                if review_rows
                else None
            ),
            "avg_rounds": sum(row.get("rounds") or 0 for row in rows) / n,
            "avg_results": sum(row.get("num_results") or 0 for row in rows) / n,
            "avg_latency_ms": sum(row["latency_ms"] for row in rows) / n,
            "errors": sum(
                1
                for row in rows
                if row.get("run_error")
                or any(item.get("error") for item in row.get("results", []))
            ),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"summary": summary, "records": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
