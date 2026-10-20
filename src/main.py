from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.graph.workflow import MASState, build_graph
from src.llm.factory import build_client
from src.tools.builtin import build_default_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the mas multi-agent workflow.")
    parser.add_argument("task", nargs="+", help="Task for the agents.")
    parser.add_argument("--backend", choices=["openai", "hf"], default=os.getenv("LLM_BACKEND", "openai"))
    parser.add_argument("--model", default=None, help="OpenAI model name (base or toolcall)")
    parser.add_argument("--base-model", default=os.getenv("HF_BASE_MODEL"))
    parser.add_argument("--adapter", default=os.getenv("HF_ADAPTER"))
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=os.getenv("HF_LOAD_IN_4BIT", "1") == "1",
    )
    parser.add_argument("--max-rounds", type=int, default=2)
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    if args.backend == "hf":
        if not args.base_model:
            parser.error("--base-model is required with --backend hf")
        client = build_client(
            "hf",
            base_model=args.base_model,
            adapter=args.adapter,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        client = build_client("openai", default_model=args.model)

    task = " ".join(args.task)
    registry = build_default_registry()
    graph = build_graph(client, registry, max_rounds=args.max_rounds)

    initial: MASState = {
        "task": task,
        "plan": [],
        "results": [],
        "review": {},
        "rounds": 0,
        "trace": [],
    }
    result = graph.invoke(initial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
