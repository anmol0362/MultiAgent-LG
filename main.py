from graph.workflow import build_graph
from utils.helpers import chat_history
from observability import observe, langfuse

app = build_graph()

@observe(name="multi-agent-run", as_type="chain")
def run_agent(query: str):
    output = app.invoke({"input": query})
    result = output["result"]

    print("\n🤖 AI:", result)

    try:
        planner_score = float(input("\nRate planner quality (0-10): "))
        routing_score = float(input("Rate routing quality (0-10): "))
        answer_score = float(input("Rate final answer quality (0-10): "))

        langfuse.score_current_trace(name="planner_quality", value=planner_score)
        langfuse.score_current_trace(name="routing_quality", value=routing_score)
        langfuse.score_current_trace(name="final_answer_quality", value=answer_score)

        print("\n✅ Scores sent to Langfuse.\n")

    except Exception as e:
        print(f"\n⚠️ Could not save scores: {e}\n")

    return output

def add_manual_scores():
    try:
        planner_score = float(input("\nRate planner quality (0-10): "))
        routing_score = float(input("Rate routing quality (0-10): "))
        answer_score = float(input("Rate final answer quality (0-10): "))

        langfuse.score_current_trace(
            name="planner_quality",
            value=planner_score
        )

        langfuse.score_current_trace(
            name="routing_quality",
            value=routing_score
        )

        langfuse.score_current_trace(
            name="final_answer_quality",
            value=answer_score
        )

        print("\n✅ Scores sent to Langfuse.\n")

    except Exception as e:
        print(f"\n⚠️ Could not save scores: {e}\n")

if __name__ == "__main__":
    while True:
        query = input("\nYou: ")

        if query.lower() == "exit":
            break

        print("\n🚀 Running...\n")

        output = run_agent(query)
        result = output["result"]

        chat_history.append(f"User: {query}")
        chat_history.append(f"AI: {result}")

    langfuse.shutdown()