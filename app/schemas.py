from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class SourceModel(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceModel]

class UploadResponse(BaseModel):
    message: str
    files_processed: int

class HistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]
