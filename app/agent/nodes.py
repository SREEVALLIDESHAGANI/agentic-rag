from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.agent.state import AgentState
from app.rag.retriever import RAGRetriever
from app.config import settings
from app.agent.tools import perform_web_search
import json

# Initialize components
llm = ChatOpenAI(model=settings.MODEL_NAME, temperature=0, api_key=settings.OPENAI_API_KEY)
retriever = RAGRetriever()

def decompose_query(state: AgentState) -> AgentState:
    """Break a complex query into sub-queries."""
    query = state["query"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert researcher. Break down the user query into 1-3 sub-queries that will help retrieve necessary information. Output as a JSON list of strings."),
        ("human", "{query}")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({"query": query})
        sub_queries = json.loads(response.content)
        if not isinstance(sub_queries, list):
            sub_queries = [query]
    except Exception:
        sub_queries = [query]
    
    return {"sub_queries": sub_queries}

def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve documents for the queries."""
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        sub_queries = [state["query"]]
    
    all_docs = []
    for q in sub_queries:
        docs = retriever.retrieve(q, top_k=3)
        for doc in docs:
            # Deduplicate by content
            if not any(d["content"] == doc["content"] for d in all_docs):
                all_docs.append(doc)
    
    return {"retrieved_docs": all_docs}

def evaluate_context(state: AgentState) -> AgentState:
    """Evaluate if retrieved context is sufficient."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        return {"needs_web_search": True}
        
    context = "\\n\\n".join([d["content"] for d in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate if the provided context contains enough information to answer the query. Respond ONLY with 'YES' or 'NO'."),
        ("human", "Query: {query}\\n\\nContext: {context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context})
    needs_search = "NO" in response.content.upper()
    
    return {"needs_web_search": needs_search}

def web_search(state: AgentState) -> AgentState:
    """Perform a web search if needed."""
    query = state["query"]
    results = perform_web_search(query)
    return {"web_results": results}

def generate_response(state: AgentState) -> AgentState:
    """Generate the final response."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    web_res = state.get("web_results", "")
    
    context_parts = []
    if docs:
        context_parts.append("Database Context:\\n" + "\\n\\n".join([d["content"] for d in docs]))
    if web_res:
        context_parts.append("Web Search Context:\\n" + web_res)
        
    context = "\\n\\n".join(context_parts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the provided context to answer the user's query. Cite your sources if possible. If the context does not contain the answer, say so."),
        ("human", "Context:\\n{context}\\n\\nQuery: {query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    
    sources = state.get("retrieved_docs", [])
    if web_res:
        sources.append({"content": "Web Search Results", "metadata": {"source": "web"}})
        
    return {"final_response": response.content, "sources": sources}

def format_output(state: AgentState) -> AgentState:
    """Format the final output."""
    # This node could restructure the data for the API response.
    # We just pass the state through as the graph ends here.
    return state
