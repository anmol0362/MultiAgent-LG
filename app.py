from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="Multi-Agent RAG API")

graph = None

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Multi-Agent RAG API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

def get_graph():
    global graph
    if graph is None:
        # ✅ correct import
        from graph.workflow import build_graph
        graph = build_graph()
    return graph

@app.post("/ask")
def ask(req: QueryRequest):
    g = get_graph()
    output = g.invoke({"input": req.query})
    return {
        "query": req.query,
        "result": output.get("result", ""),
        "step_results": output.get("step_results", [])
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)