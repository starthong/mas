from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.executor import Executor, results_to_context
from src.agents.planner import Planner
from src.agents.reviewer import Reviewer
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry


class MASState(TypedDict, total=False):
    task: str
    plan: list[dict[str, Any]]
    results: list[dict[str, Any]]
    review: dict[str, Any]
    rounds: int
    trace: list[str]


def build_graph(
    client: LLMClient,
    registry: ToolRegistry,
    max_rounds: int = 2,
):
    planner = Planner(client)
    executor = Executor(client, registry)
    reviewer = Reviewer(client)

    def planner_node(state: MASState) -> dict[str, Any]:
        review = state.get("review") or {}
        critique = str(review.get("suggestion", "")) if state.get("rounds", 0) else ""
        plan = planner.plan(state["task"], critique=critique)
        trace = list(state.get("trace", []))
        trace.append(f"planner: {len(plan)} subtask(s)")
        return {"plan": plan, "results": [], "trace": trace}

    def executor_node(state: MASState) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        context = ""
        for subtask in state.get("plan", []):
            result = executor.run(state["task"], subtask, context=context)
            results.append(result)
            context = results_to_context(results)

        trace = list(state.get("trace", []))
        trace.append(f"executor: {len(results)} result(s)")
        return {"results": results, "trace": trace}

    def reviewer_node(state: MASState) -> dict[str, Any]:
        review = reviewer.review(
            state["task"],
            state.get("plan", []),
            state.get("results", []),
        )
        rounds = state.get("rounds", 0) + 1
        trace = list(state.get("trace", []))
        trace.append(
            f"reviewer: pass={review.get('pass')} score={review.get('score')} "
            f"round={rounds}"
        )
        return {"review": review, "rounds": rounds, "trace": trace}

    def route(state: MASState) -> str:
        review = state.get("review") or {}
        if review.get("pass"):
            return "end"
        if state.get("rounds", 0) >= max_rounds:
            return "end"
        return "replan"

    graph = StateGraph(MASState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route,
        {"replan": "planner", "end": END},
    )
    return graph.compile()
