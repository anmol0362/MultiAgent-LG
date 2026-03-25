from config import llm
from prompts import TOOL_DECISION_PROMPT, FINAL_SYNTHESIS_PROMPT, REASONING_PROMPT
from utils.parser import parse_steps
from utils.logger import log_event
from agents.researcher import researcher
from agents.coder import coder
from observability import observe, langfuse

@observe(name="executor-agent", as_type="agent")
def executor(state):
    query = state["input"]
    plan = state["plan"]

    log_event("[EXECUTOR]", "Executing step-by-step...")

    steps = parse_steps(plan)
    if not steps:
        steps = [query]

    results = []

    for step in steps:
        log_event("Step", step)
        langfuse.update_current_span(
    metadata={"current_step": step}
)

        tool_decision = llm.invoke(
    TOOL_DECISION_PROMPT.format(step=step)
).content.strip().upper()

# hard guardrail against unnecessary coding
    step_lower = step.lower()

    if any(keyword in step_lower for keyword in ["compare", "vs", "versus", "tradeoff", "pros", "cons", "better"]):
        tool_decision = "COMPARE"
    elif any(keyword in step_lower for keyword in ["plan", "roadmap", "recommend", "advice", "should i", "learning", "learn", "study", "realistic plan", "focus on"]):
        tool_decision = "ADVISE"
    elif any(keyword in step_lower for keyword in ["explain", "summarize", "analyze", "understand", "why", "how"]):
        tool_decision = "REASON"

        log_event("Tool Decision", tool_decision)
        langfuse.update_current_span(
    metadata={"tool_decision": tool_decision}
)

        if tool_decision == "SEARCH":
            step_result = researcher(step, use_search=True)

        elif tool_decision == "CODE":
            step_result = coder(step)

        elif tool_decision == "COMPARE":
            step_result = {
        "agent": "comparer",
        "status": "success",
        "tool_used": "none",
        "output": llm.invoke(REASONING_PROMPT.format(query=f"Compare these options clearly and practically: {step}")).content
    }

    elif tool_decision == "ADVISE":
        step_result = {
        "agent": "advisor",
        "status": "success",
        "tool_used": "none",
        "output": llm.invoke(REASONING_PROMPT.format(query=f"Give practical advice and a clear recommendation: {step}")).content
    }

    else:  # REASON
        step_result = {
        "agent": "reasoner",
        "status": "success",
        "tool_used": "none",
        "output": llm.invoke(REASONING_PROMPT.format(query=step)).content
    }
        results.append(f"{step} -> {step_result['output']}")

    results_text = "\n".join(results)

    final = llm.invoke(
        FINAL_SYNTHESIS_PROMPT.format(query=query, results=results_text)
    )

    return {
        "result": final.content,
        "input": query,
        "step_results": results
    }