# AgentRAG — Autonomous Research & Knowledge Agent

An agentic RAG system that autonomously decomposes user queries, retrieves relevant information, evaluates retrieved context, and generates grounded responses using an LLM.

## Architecture Diagram

```
User Query --> API (FastAPI) --> Agent Graph (LangGraph)
                                      |
                                      +--> Decompose Query
                                      |
                                      +--> Retrieve Docs (ChromaDB)
                                      |
                                      +--> Evaluate Context
                                      |    |
                                      |    +--> Web Search (if needed)
                                      |
                                      +--> Generate Response (LLM)
                                      |
                                      +--> Format Output
                                      |
Response <-----------------------------+
```

## Features
- Agentic RAG system with autonomous query decomposition
- Semantic search and document retrieval using embeddings and ChromaDB
- Agent workflow with dynamic tool calling (retrieval, web search, reasoning)
- Prompt engineering, conversation memory, context management
- Structured JSON outputs

## Tech Stack
- Python 3.10+
- OpenAI API
- LangGraph & LangChain
- RAG (Retrieval-Augmented Generation)
- Embeddings
- Vector Database (ChromaDB)
- FastAPI Backend

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd agentic-rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Copy `.env.example` to `.env` and add your keys.
   ```bash
   cp .env.example .env
   ```

## Usage

1. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

2. Access Swagger UI at `http://localhost:8000/docs` to interact with the endpoints.

### API Endpoints
- `POST /api/query`: Submit a query to the agent
- `POST /api/upload`: Upload documents for the knowledge base
- `GET /api/history`: Get conversation history
- `DELETE /api/history`: Clear conversation history

## Project Structure
```
agentic-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py              # FastAPI entry point
├── app/
│   ├── __init__.py
│   ├── config.py        # Settings and env config
│   ├── schemas.py       # Pydantic models
│   ├── agent/           # LangGraph agent definitions
│   ├── rag/             # Document loading, embedding, retrieval
│   └── api/             # FastAPI routes
```
