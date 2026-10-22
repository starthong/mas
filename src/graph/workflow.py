from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.executor import Executor, results_to_context
from src.agents.planner import Planner
from src.agents.reviewer import Reviewer
from src.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from src.llm.client import LLMClient


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
    temperature: float = 0.0,
    planner_client: Any = None,
    reviewer_client: Any = None,
    enable_reviewer: bool = True,
):
    planner = Planner(planner_client or client, temperature=temperature)
    executor = Executor(client, registry, temperature=temperature)
    reviewer = Reviewer(reviewer_client or client, temperature=temperature)

    def planner_node(state: MASState) -> dict[str, Any]:
        review = state.get("review") or {}
        critique = str(review.get("suggestion", "")) if state.get("rounds", 0) else ""
        plan = planner.plan(state["task"], critique=critique)
        trace = list(state.get("trace", []))
        if planner.last_error:
            trace.append(f"planner_error: {planner.last_error}")
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

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")

    if enable_reviewer:
        graph.add_node("reviewer", reviewer_node)
        graph.add_edge("executor", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            route,
            {"replan": "planner", "end": END},
        )
    else:
        graph.add_edge("executor", END)

    return graph.compile()
