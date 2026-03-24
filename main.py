import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import AzureChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

load_dotenv()
chat_history = []

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

    print("\n🤖 [EXECUTOR AGENT]")

    history_text = "\n".join(chat_history)

    # decide tool
    decision = llm.invoke(f"""
Conversation:
{history_text}

Task: {query}

Should I use search? YES or NO
""").content

    print("🤔 Tool Decision:", decision)

    if "YES" in decision.upper():
        print("🌐 Using SEARCH TOOL...\n")
        tool_data = search_tool.invoke(query)
    else:
        tool_data = "No external data used."

    result = llm.invoke(f"""
Conversation:
{history_text}

Task: {query}

Plan:
{plan}

Data:
{tool_data}

Give final answer.
""")

    print("✅ Done\n")

    return {
        "result": result.content,
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