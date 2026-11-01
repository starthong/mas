"""在系统任务上运行单 Agent 和多智能体实验。"""
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
    text = str(result.get("result") or "").strip()
    if not text or "暂无数据" in text:
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


def _usage_snapshot(clients: list[Any]) -> dict[str, int]:
    total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for client in clients:
        usage = getattr(client, "usage", {}) or {}
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before.get(key, 0) for key in before}


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[index]


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
    mode: str = "sequential",
    planner_few_shot: bool = False,
    max_rounds: int = 2,
    planner_samples: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = build_graph(
        client,
        registry,
        max_rounds=max_rounds,
        temperature=temperature,
        planner_client=planner_client,
        reviewer_client=reviewer_client,
        enable_reviewer=enable_reviewer,
        mode=mode,
        planner_few_shot=planner_few_shot,
        planner_samples=planner_samples,
    )
    state = graph.invoke(
        {
            "task": task["task"],
            "plan": [],
            "results": [],
            "review": {},
            "rounds": 0,
            "cursor": 0,
            "trace": [],
        }
    )
    return state.get("results", []), state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/system_tasks.jsonl")
    parser.add_argument("--modes", default="single,sequential")
    parser.add_argument("--categories", default=None)
    parser.add_argument("--planner-examples", action="store_true")
    parser.add_argument("--planner-samples", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperatures", default="0")
    parser.add_argument("--backend", choices=["openai", "hf"], default="hf")
    parser.add_argument("--model", default="toolcall")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--planner-adapter", default=None)
    parser.add_argument("--reviewer-adapter", default=None)
    parser.add_argument("--planner-base-model", default=None)
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
    categories = {
        item.strip() for item in (args.categories or "").split(",") if item.strip()
    }
    if categories:
        tasks = [task for task in tasks if task.get("category") in categories]
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
    if reviewer_mode == "base":
        if args.backend != "hf":
            raise SystemExit("base reviewer is only supported with --backend hf")
        base_client = build_client(
            "hf",
            base_model=args.base_model,
            adapter=None,
            load_in_4bit=args.load_in_4bit,
        )

    planner_base_model = args.planner_base_model or args.base_model
    planner_client = None
    reviewer_client = base_client if reviewer_mode == "base" else None
    extra_clients = []
    if planner_mode == "base":
        if args.backend != "hf":
            raise SystemExit("base planner is only supported with --backend hf")
        planner_client = build_client(
            "hf",
            base_model=planner_base_model,
            adapter=None,
            load_in_4bit=args.load_in_4bit,
        )
        extra_clients.append(planner_client)
    elif args.planner_adapter:
        planner_client = build_client(
            "hf",
            base_model=planner_base_model,
            adapter=args.planner_adapter,
            load_in_4bit=args.load_in_4bit,
        )
        extra_clients.append(planner_client)
    if reviewer_mode == "lora" and args.reviewer_adapter:
        reviewer_client = build_client(
            "hf",
            base_model=args.base_model,
            adapter=args.reviewer_adapter,
            load_in_4bit=args.load_in_4bit,
        )
        extra_clients.append(reviewer_client)
    enable_reviewer = reviewer_mode != "none"
    usage_clients = [client] + ([base_client] if base_client is not None else []) + extra_clients

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.jsonl")
    partial_handle = partial_path.open("w", encoding="utf-8")
    total_runs = len(modes) * len(temperatures) * len(tasks) * args.runs
    done = 0
    print(f"checkpoint: {partial_path}", flush=True)

    records: list[dict[str, Any]] = []
    for mode in modes:
        for temperature in temperatures:
            for task in tasks:
                for run_index in range(1, args.runs + 1):
                    usage_before = _usage_snapshot(usage_clients)
                    started = time.perf_counter()
                    run_error: str | None = None
                    try:
                        if mode == "single":
                            results, state = run_single(
                                client, registry, task, temperature
                            )
                        elif mode in ("sequential", "supervisor", "debate"):
                            results, state = run_sequential(
                                client,
                                registry,
                                task,
                                temperature,
                                planner_client=planner_client,
                                reviewer_client=reviewer_client,
                                enable_reviewer=enable_reviewer,
                                mode=mode,
                                planner_few_shot=args.planner_examples,
                                max_rounds=args.max_rounds,
                                planner_samples=args.planner_samples,
                            )
                        else:
                            raise SystemExit(f"unknown mode: {mode}")
                    except Exception as exc:  # noqa: BLE001 - one task must not abort the run
                        results = []
                        state = {}
                        run_error = f"{type(exc).__name__}: {exc}"
                    latency_ms = (time.perf_counter() - started) * 1000
                    usage = _usage_delta(
                        usage_before, _usage_snapshot(usage_clients)
                    )

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

                    record = {
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
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "llm_calls": usage["calls"],
                        "results": results,
                        "trace": state.get("trace", []),
                        "review": review,
                        "run_error": run_error,
                    }
                    records.append(record)
                    partial_handle.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    partial_handle.flush()
                    done += 1
                    print(
                        f"[{done}/{total_runs}] {mode}@{temperature} "
                        f"{task.get('id')} run={run_index} "
                        f"success={success} latency={latency_ms:.0f}ms",
                        flush=True,
                    )

    partial_handle.close()

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
            "p95_latency_ms": _p95([row["latency_ms"] for row in rows]),
            "avg_prompt_tokens": sum(row.get("prompt_tokens", 0) for row in rows) / n,
            "avg_completion_tokens": (
                sum(row.get("completion_tokens", 0) for row in rows) / n
            ),
            "avg_llm_calls": sum(row.get("llm_calls", 0) for row in rows) / n,
            "errors": sum(
                1
                for row in rows
                if row.get("run_error")
                or any(item.get("error") for item in row.get("results", []))
            ),
        }

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
