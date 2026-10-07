from langchain_openai import OpenAIEmbeddings
from app.config import settings

def get_embeddings():
    """Get the OpenAI embeddings model."""
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY
    )
