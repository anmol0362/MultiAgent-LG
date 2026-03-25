from config import llm
from prompts import PLANNER_PROMPT
from utils.logger import log_event
from utils.helpers import chat_history
from observability import observe

@observe(name="planner-agent", as_type="agent")
def planner(state):
    query = state["input"]

    log_event("[PLANNER AGENT]", f"Task: {query}")

    history_text = "\n".join(chat_history)

    raw_plan = llm.invoke(
    f"{PLANNER_PROMPT}\n\nConversation so far:\n{history_text}\n\nNew user query:\n{query}"
).content

# keep only first 4 non-empty lines
    plan_lines = [line.strip() for line in raw_plan.split("\n") if line.strip()]
    plan = "\n".join(plan_lines[:4])

    log_event("Plan", plan)

    return {
        "plan": plan,
        "input": query
    }