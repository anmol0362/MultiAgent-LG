from config import llm
from observability import observe

# temporary simple RAG agent
# later you can connect Pinecone / Chroma / FAISS

@observe(name="rag-agent", as_type="agent")
def rag_agent(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    response = llm.invoke(f"""
You are a retrieval-augmented research agent.

User query:
{query}

Retrieved context:
{context}

Your job:
- use ONLY the retrieved context if possible
- summarize the useful facts
- return grounded information
- if context is weak, say what is missing

Return a clean helpful answer.
""")

    return {
        "agent": "rag_agent",
        "status": "success",
        "tool_used": "retrieval",
        "output": response.content
    }