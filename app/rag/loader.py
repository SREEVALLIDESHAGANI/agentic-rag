from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.vectorstore import get_vectorstore
import os

def load_and_process_document(file_path: str):
    """Load a document, chunk it, and store in vector database."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
        
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        vectorstore.persist()
        
    return len(chunks)
