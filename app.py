from fastapi import FastAPI
from pydantic import BaseModel
from graph.workflow import build_graph

app = FastAPI(title="Multi-Agent RAG API")
graph = build_graph()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Multi-Agent RAG API is running 🚀"}

@app.post("/ask")
def ask(req: QueryRequest):
    output = graph.invoke({"input": req.query})
    return {
        "query": req.query,
        "result": output["result"],
        "step_results": output.get("step_results", [])
    }