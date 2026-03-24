import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import AzureChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

from dotenv import load_dotenv
load_dotenv()
chat_history = []
import os
print("LangSmith Key:", os.getenv("LANGCHAIN_API_KEY"))

# ------------------------
# LLM
# ------------------------


llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    deployment_name="gpt-4o"
)

# ------------------------
# TOOLS
# ------------------------

search_tool = TavilySearchResults()
python_tool = PythonREPLTool()

# ------------------------
# AGENTS
# ------------------------

def researcher(state):
    query = state["input"]

    print("\n🟡 [Researcher Agent]")
    print("🔍 Query:", query)

    # Step 1: Decide if search needed
    decision = llm.invoke(f"""
You are a smart research agent.

User query: {query}

Should you use web search to answer this?
Answer only YES or NO.
""").content

    print("🤔 Decision:", decision)

    if "YES" in decision.upper():
        print("🌐 Using SEARCH TOOL...\n")
        tool_data = search_tool.invoke(query)
    else:
        tool_data = "No external search used."

    # Step 2: Final response
    final = llm.invoke(f"""
Answer the user properly.

Query: {query}

Data:
{tool_data}

Give a clean, helpful answer.
""")

    print("✅ Researcher done\n")

    return {
        "result": final.content,
        "input": query
    }


def coder(state):
    query = state["input"]

    print("\n🟣 [Coder Agent]")
    print("💻 Task:", query)

    # Step 1: Generate code
    code = llm.invoke(f"""
Write clean Python code for this:

{query}

Only return code, no explanation.
""").content

    print("\n⚡ Generated Code:\n", code)

    # Step 2: Execute code
    try:
        print("\n▶️ Executing code...")
        output = python_tool.invoke(code)
    except Exception as e:
        output = str(e)

    print("✅ Coder done\n")

    return {
        "result": f"Code:\n{code}\n\nOutput:\n{output}",
        "input": query
    }

# ------------------------
# SUPERVISOR
# ------------------------

def supervisor(state):
    query = state["input"]

    print("\n🧠 [Supervisor]")
    print("User:", query)

    # smarter routing
    decision = llm.invoke(f"""
Classify this query:

"{query}"

Return ONLY one word:
- CODER → if coding / python / technical execution
- RESEARCHER → for explanation / info / trends
""").content.strip().upper()

    print("📌 Decision:", decision)

    if "CODER" in decision:
        print("➡️ Routing to CODER\n")
        return {"next": "coder", "input": query}
    else:
        print("➡️ Routing to RESEARCHER\n")
        return {"next": "researcher", "input": query}
    
def planner(state):
    query = state["input"]

    print("\n🧠 [PLANNER AGENT]")
    print("📌 Task:", query)

    history_text = "\n".join(chat_history)

    plan = llm.invoke(f"""
You are a planning agent.

Conversation so far:
{history_text}

New user query:
{query}

Break this into steps.
""").content

    print("\n📝 Plan:\n", plan)

    return {
        "plan": plan,
        "input": query
    }

def executor(state):
    query = state["input"]
    plan = state["plan"]

    print("\n🤖 [EXECUTOR - PRO MODE]")
    print("📌 Executing step-by-step...\n")

    steps = []
    for s in plan.split("\n"):
        s = s.strip()
        if s.lower().startswith("step"):
            steps.append(s)

    steps = steps[:5]

    if not steps:
        steps = [query]

    results = []

    for step in steps:
        if not step.strip():
            continue

        # ✅ filter garbage steps
        if any(x in step.lower() for x in ["substep", "---", "here’s", "would you"]):
            continue

        print(f"\n➡️ Step: {step}")

        # ✅ SINGLE tool decision (FIXED)
        if "find" in step.lower() or "research" in step.lower():
            tool_decision = "SEARCH"
        else:
            tool_decision = llm.invoke(f"""
Step: {step}

Does this step require:
- SEARCH
- CODE
- NONE

Answer only one word.
""").content.strip().upper()

        print("🛠️ Tool Decision:", tool_decision)

        # 🔍 SEARCH
        if "SEARCH" in tool_decision:
            print("🌐 Using SEARCH TOOL...")
            tool_output = search_tool.invoke(step)

        # 💻 CODE
        elif "CODE" in tool_decision:
            print("💻 Generating + Running Code...")
            code = llm.invoke(f"""
Return ONLY valid Python code.
No explanation.
No markdown.

Task: {step}
""").content

            try:
                tool_output = python_tool.invoke(code)
            except Exception as e:
                tool_output = str(e)

        # 🧠 NORMAL
        else:
            tool_output = llm.invoke(f"Solve this step: {step}").content

        print("✅ Step Result:", str(tool_output)[:200], "...")

        results.append(f"{step} → {tool_output}")

    # ✅ prevent empty results
    if not results:
        results.append("No valid steps executed.")

    final = llm.invoke(f"""
Task: {query}

Step Results:
{results}

Use ONLY these results.
Do NOT say you cannot answer.
Give final structured answer.
""")

    print("\n🎯 FINAL DONE\n")

    return {
        "result": final.content,
        "input": query
    }
# ------------------------
# GRAPH
# ------------------------

builder = StateGraph(dict)

builder.add_node("planner", planner)
builder.add_node("executor", executor)

builder.set_entry_point("planner")

builder.add_edge("planner", "executor")
builder.add_edge("executor", "__end__")

app = builder.compile()

# ------------------------
# RUN
# ------------------------

if __name__ == "__main__":
    while True:
        query = input("\nYou: ")

        if query.lower() == "exit":
            break

        print("\n🚀 Running...\n")

        output = app.invoke({"input": query})

        result = output["result"]

        print("\n🤖 AI:", result)

        # 🔥 STORE MEMORY
        chat_history.append(f"User: {query}")
        chat_history.append(f"AI: {result}")