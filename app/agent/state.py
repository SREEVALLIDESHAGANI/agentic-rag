import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Represents the state of our agent."""
    query: str
    sub_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    needs_web_search: bool
    web_results: str
    final_response: str
    sources: List[Dict[str, Any]]
    messages: Annotated[List[BaseMessage], operator.add]
