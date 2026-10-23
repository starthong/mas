from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.debate import DebatePlanner
from src.agents.executor import Executor, results_to_context
from src.agents.planner import Planner
from src.agents.reviewer import Reviewer
from src.agents.supervisor import Supervisor
from src.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class MASState(TypedDict, total=False):
    task: str
    plan: list[dict[str, Any]]
    results: list[dict[str, Any]]
    review: dict[str, Any]
    rounds: int
    cursor: int
    supervisor_action: dict[str, Any]
    trace: list[str]


def build_graph(
    client: LLMClient,
    registry: ToolRegistry,
    max_rounds: int = 2,
    temperature: float = 0.0,
    planner_client: Any = None,
    reviewer_client: Any = None,
    enable_reviewer: bool = True,
    mode: str = "sequential",
    planner_few_shot: bool = False,
):
    planner = Planner(
        planner_client or client,
        temperature=temperature,
        few_shot=planner_few_shot,
    )
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
        return {"plan": plan, "results": [], "cursor": 0, "trace": trace}

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

    def route_review(state: MASState) -> str:
        review = state.get("review") or {}
        if review.get("pass"):
            return "end"
        if state.get("rounds", 0) >= max_rounds:
            return "end"
        return "replan"

    if mode == "sequential":
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
                route_review,
                {"replan": "planner", "end": END},
            )
        else:
            graph.add_edge("executor", END)
        return graph.compile()

    if mode == "supervisor":
        supervisor = Supervisor(client, temperature=temperature)

        def supervisor_node(state: MASState) -> dict[str, Any]:
            decision = supervisor.decide(
                state["task"],
                state.get("plan", []),
                state.get("results", []),
                state.get("cursor", 0),
                state.get("rounds", 0),
            )
            trace = list(state.get("trace", []))
            if supervisor.last_error:
                trace.append(f"supervisor_error: {supervisor.last_error}")
            trace.append(
                f"supervisor: action={decision['action']} "
                f"subtask={decision.get('subtask_id')} {decision.get('reason', '')}"
            )
            return {"supervisor_action": decision, "trace": trace}

        def executor_one_node(state: MASState) -> dict[str, Any]:
            plan = state.get("plan", [])
            cursor = state.get("cursor", 0)
            trace = list(state.get("trace", []))
            if cursor >= len(plan):
                trace.append("executor: no subtask left")
                return {"trace": trace}

            subtask = plan[cursor]
            results = list(state.get("results", []))
            result = executor.run(
                state["task"],
                subtask,
                context=results_to_context(results),
            )
            results.append(result)
            trace.append(
                f"executor: subtask {subtask.get('id')} -> "
                f"{result.get('tool')} error={result.get('error')}"
            )
            return {"results": results, "cursor": cursor + 1, "trace": trace}

        def route_supervisor(state: MASState) -> str:
            action = (state.get("supervisor_action") or {}).get("action", "finish")
            if action == "execute":
                return "execute"
            if action == "review" and enable_reviewer:
                return "review"
            return "end"

        graph = StateGraph(MASState)
        graph.add_node("planner", planner_node)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("executor", executor_one_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "supervisor")
        if enable_reviewer:
            graph.add_node("reviewer", reviewer_node)
            graph.add_conditional_edges(
                "supervisor",
                route_supervisor,
                {"execute": "executor", "review": "reviewer", "end": END},
            )
            graph.add_edge("executor", "supervisor")
            graph.add_conditional_edges(
                "reviewer",
                route_review,
                {"replan": "planner", "end": END},
            )
        else:
            graph.add_conditional_edges(
                "supervisor",
                route_supervisor,
                {"execute": "executor", "review": END, "end": END},
            )
            graph.add_edge("executor", "supervisor")
        return graph.compile()

    if mode == "debate":
        debate_planner = DebatePlanner(
            planner_client or client,
            temperature=temperature,
            few_shot=planner_few_shot,
        )

        def debate_planner_node(state: MASState) -> dict[str, Any]:
            review = state.get("review") or {}
            critique = (
                str(review.get("suggestion", "")) if state.get("rounds", 0) else ""
            )
            plan = debate_planner.plan(state["task"], critique=critique)
            trace = list(state.get("trace", []))
            if debate_planner.last_error:
                trace.append(f"judge_error: {debate_planner.last_error}")
            trace.append(
                f"debate: choice={debate_planner.last_choice} "
                f"{debate_planner.last_reason} -> {len(plan)} subtask(s)"
            )
            return {"plan": plan, "results": [], "cursor": 0, "trace": trace}

        graph = StateGraph(MASState)
        graph.add_node("debate_planner", debate_planner_node)
        graph.add_node("executor", executor_node)
        graph.add_edge(START, "debate_planner")
        graph.add_edge("debate_planner", "executor")
        if enable_reviewer:
            graph.add_node("reviewer", reviewer_node)
            graph.add_edge("executor", "reviewer")
            graph.add_conditional_edges(
                "reviewer",
                route_review,
                {"replan": "debate_planner", "end": END},
            )
        else:
            graph.add_edge("executor", END)
        return graph.compile()

    raise ValueError(f"unknown mode: {mode}")
