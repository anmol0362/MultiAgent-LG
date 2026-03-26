from langgraph.graph import StateGraph, END
from state import AgentState
from agents.planner import planner
from agents.executor import executor
from agents.reviewer import reviewer


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("reviewer", reviewer)

    builder.set_entry_point("planner")

    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reviewer")
    builder.add_edge("reviewer", END)

    return builder.compile()