from typing import Dict, TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    decompose_query,
    retrieve_documents,
    evaluate_context,
    web_search,
    generate_response,
    format_output
)

def build_graph() -> StateGraph:
    """Build the LangGraph workflow for the agent."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("evaluate_context", evaluate_context)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("format_output", format_output)

    # Set entry point
    workflow.set_entry_point("decompose_query")

    # Add edges
    workflow.add_edge("decompose_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "evaluate_context")
    
    # Conditional edge based on context evaluation
    workflow.add_conditional_edges(
        "evaluate_context",
        lambda state: "web_search" if state.get("needs_web_search") else "generate_response",
        {
            "web_search": "web_search",
            "generate_response": "generate_response"
        }
    )
    
    workflow.add_edge("web_search", "generate_response")
    workflow.add_edge("generate_response", "format_output")
    workflow.add_edge("format_output", END)

    return workflow.compile()
