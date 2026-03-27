graph = None

def get_graph():
    global graph
    if graph is None:
        graph = build_graph()
    return graph

@app.post("/ask")
def ask(req: QueryRequest):
    g = get_graph()
    output = g.invoke({"input": req.query})
    return {
        "query": req.query,
        "result": output["result"],
        "step_results": output.get("step_results", [])
    }