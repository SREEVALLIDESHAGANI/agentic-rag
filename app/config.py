import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4-turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    UPLOAD_DIR: str = "./uploaded_docs"

    class Config:
        env_file = ".env"

settings = Settings()
