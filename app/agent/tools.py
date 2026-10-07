from langchain_community.tools import DuckDuckGoSearchResults

def perform_web_search(query: str, max_results: int = 3) -> str:
    """Perform a web search using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchResults(num_results=max_results)
        return search.invoke(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"
