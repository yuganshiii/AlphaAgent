"""
LangGraph graph definition for AlphaAgent.

Flow:
  START → planner → tool_executor → synthesizer → critic
                                                      ↓
                               (score < threshold AND iteration < max) → planner
                                                      ↓
                                              (otherwise) → END
"""
from langgraph.graph import StateGraph, END

from src.agent.state import GraphState
from src.agent.nodes.planner import planner_node
from src.agent.nodes.tool_executor import tool_executor_node
from src.agent.nodes.synthesizer import synthesizer_node
from src.agent.nodes.critic import critic_node
from src.config import settings


def _should_loop(state: GraphState) -> str:
    score = state.get("critique_score", 1.0)
    iteration = state.get("iteration", 0)
    if score < settings.CRITIQUE_THRESHOLD and iteration < settings.MAX_ITERATIONS:
        return "planner"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool_executor")
    graph.add_edge("tool_executor", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges("critic", _should_loop, {"planner": "planner", END: END})

    return graph.compile()


# Singleton — import this in API and scripts
agent_graph = build_graph()
