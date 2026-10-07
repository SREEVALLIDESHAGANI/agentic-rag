from typing import List, Dict, Any
from app.rag.vectorstore import get_vectorstore

class RAGRetriever:
    def __init__(self):
        self.vectorstore = get_vectorstore()
        
    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve documents based on query."""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
            
        return formatted_results
