import os
import sys
from pathlib import Path

# Add your AGENTIC AI project path
AGENTIC_AI_PATH = r"D:\AGENTIC AI"
sys.path.append(AGENTIC_AI_PATH)

from src.vectorstore import FaissVectorStore

# Load FAISS store once
store = FaissVectorStore(
    persist_dir=os.path.join(AGENTIC_AI_PATH, "faiss_store"),
    deployment_name="text-embedding-3-small"
)

store.load()


def rag_search(query: str, top_k: int = 3):
    """
    Search your FAISS RAG and return top matching chunks.
    """
    results = store.query(query, top_k=top_k)

    docs = []
    for r in results:
        if r["metadata"] and "text" in r["metadata"]:
            docs.append(r["metadata"]["text"])

    return docs