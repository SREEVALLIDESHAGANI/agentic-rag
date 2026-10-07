from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.schemas import QueryRequest, QueryResponse, UploadResponse, HistoryResponse
from app.agent.graph import build_graph
from app.rag.loader import load_and_process_document
from app.config import settings
import os
import shutil

router = APIRouter()

# In-memory session history
session_history = {}
graph = build_graph()

@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Submit a query to the agent."""
    session_id = request.session_id
    
    if session_id not in session_history:
        session_history[session_id] = []
        
    # Append user query to history
    session_history[session_id].append({"role": "user", "content": request.query})
    
    # Run agent graph
    state = {"query": request.query}
    try:
        result_state = graph.invoke(state)
        answer = result_state.get("final_response", "I could not find an answer.")
        sources = result_state.get("sources", [])
        
        # Append agent response to history
        session_history[session_id].append({"role": "agent", "content": answer})
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents for the knowledge base."""
    total_chunks = 0
    
    for file in files:
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            chunks = load_and_process_document(file_path)
            total_chunks += chunks
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing {file.filename}: {str(e)}")
            
    return UploadResponse(message="Upload and processing successful", files_processed=len(files))

@router.get("/history", response_model=HistoryResponse)
async def get_history(session_id: str = "default"):
    """Get conversation history."""
    history = session_history.get(session_id, [])
    return HistoryResponse(session_id=session_id, history=history)

@router.delete("/history")
async def clear_history(session_id: str = "default"):
    """Clear conversation history."""
    if session_id in session_history:
        del session_history[session_id]
    return {"message": f"History for session {session_id} cleared."}
