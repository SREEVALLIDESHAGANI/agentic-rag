from langchain_community.vectorstores import Chroma
from app.rag.embeddings import get_embeddings
from app.config import settings
import os

def get_vectorstore():
    """Initialize or get the ChromaDB vector store."""
    embeddings = get_embeddings()
    os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
    return Chroma(
        persist_directory=settings.CHROMA_DB_DIR,
        embedding_function=embeddings
    )
