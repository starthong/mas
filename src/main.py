from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.graph.workflow import MASState, build_graph
from src.llm.client import LLMClient
from src.tools.builtin import build_default_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the mas multi-agent workflow.")
    parser.add_argument("task", nargs="+", help="Task for the agents.")
    parser.add_argument("--model", default=None, help="base or toolcall")
    parser.add_argument("--max-rounds", type=int, default=2)
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    task = " ".join(args.task)
    client = LLMClient(default_model=args.model)
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
