from __future__ import annotations

import json

from src.graph.workflow import build_graph
from src.tools.builtin import build_default_registry


class FakeClient:
    def chat(self, messages, temperature=0.0, max_tokens=512):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "judge" in system.lower():
            return json.dumps({"choice": 1, "reason": "plan 1"})
        if "supervisor" in system.lower():
            return "not a json object"
        if "reviewer" in system.lower():
            return json.dumps(
                {"pass": True, "score": 1.0, "issues": [], "suggestion": ""}
            )
        if "critic planner" in system.lower():
            return json.dumps(
                {
                    "subtasks": [
                        {"id": 1, "goal": "first", "expected_output": "a"},
                        {"id": 2, "goal": "second", "expected_output": "b"},
                    ]
                }
            )
        if "planner" in system.lower():
            return json.dumps(
                {
                    "subtasks": [
                        {"id": 1, "goal": "first", "expected_output": "a"},
                        {"id": 2, "goal": "second", "expected_output": "b"},
                    ]
                }
            )
        if "tool-calling agent" in system.lower():
            return json.dumps(
                {"name": "get_weather", "arguments": {"city": "北京"}}
            )
        raise AssertionError((system[:40], user[:40]))


def run_mode(mode: str):
    graph = build_graph(
        FakeClient(),
        build_default_registry(),
        mode=mode,
        planner_few_shot=True,
    )
    return graph.invoke(
        {
            "task": "multi step task",
            "plan": [],
            "results": [],
            "review": {},
            "rounds": 0,
            "cursor": 0,
            "trace": [],
        }
    )


def test_sequential_runs() -> None:
    state = run_mode("sequential")
    assert len(state["results"]) == 2
    assert state["review"]["pass"] is True


def test_supervisor_runs() -> None:
    state = run_mode("supervisor")
    assert len(state["results"]) == 2
    assert state["cursor"] == 2
    assert any("supervisor:" in item for item in state["trace"])


def test_debate_runs() -> None:
    state = run_mode("debate")
    assert len(state["results"]) == 2
    assert any("debate:" in item for item in state["trace"])
