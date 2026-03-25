from langgraph.graph import StateGraph
from state import AgentState
from agents.planner import planner
from agents.executor import executor

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner)
    builder.add_node("executor", executor)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "__end__")

    return builder.compile()