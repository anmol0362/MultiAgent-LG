from tools.rag_tool import rag_search

query = "What are monitoring skills in aviation?"

results = rag_search(query)

print("\n🔍 RAG Results:\n")
for i, r in enumerate(results, 1):
    print(f"Result {i}:\n{r[:500]}\n")