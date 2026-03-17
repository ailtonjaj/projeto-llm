from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import asyncio

class AgentState(TypedDict):
    query: str
    user_skills: List[str]
    target_role: str
    intent: str
    retrieved_docs: list
    answer: str
    citations: list
    self_check_passed: bool
    retry_count: int
    skill_gap: dict
    learning_path: dict
    safety_disclaimer: str

def build_graph():
    from src.agents.supervisor import supervisor_node
    from src.agents.retriever import retriever_node
    from src.agents.safety import safety_node
    from src.agents.answerer import answerer_node
    from src.agents.self_check import self_check_node
    from src.agents.automation.skill_gap_analyzer import skill_gap_node
    from src.agents.automation.learning_path import learning_path_node

    def skill_gap_node_sync(state: dict) -> dict:
        return asyncio.run(skill_gap_node(state))

    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("retriever", retriever_node)
    g.add_node("safety", safety_node)
    g.add_node("answerer", answerer_node)
    g.add_node("self_check", self_check_node)
    g.add_node("skill_gap", skill_gap_node_sync)
    g.add_node("learning_path", learning_path_node)

    g.set_entry_point("supervisor")

    def route_intent(state):
        if state.get("intent") == "automation":
            return "skill_gap"
        elif state.get("intent") == "refuse":
            return END
        return "retriever"

    def route_self_check(state):
        if state.get("self_check_passed"):
            return "safety"
        if state.get("retry_count", 0) < 1:
            return "retriever"
        return END

    g.add_conditional_edges("supervisor", route_intent, {
        "retriever": "retriever",
        "skill_gap": "skill_gap",
        END: END
    })
    g.add_edge("retriever", "answerer")
    g.add_edge("answerer", "self_check")
    g.add_conditional_edges("self_check", route_self_check, {
        "safety": "safety",
        "retriever": "retriever",
        END: END
    })
    g.add_edge("safety", END)
    g.add_edge("skill_gap", "learning_path")
    g.add_edge("learning_path", END)

    return g.compile()
