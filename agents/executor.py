from config import llm
from prompts import TOOL_DECISION_PROMPT, REASONING_PROMPT
from utils.parser import parse_steps
from utils.logger import log_event
from agents.researcher import researcher
from agents.coder import coder
from observability import observe, langfuse
from agents.rag_agent import rag_agent



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

        # Hard guardrails against bad routing
        step_lower = step.lower()

        if any(keyword in step_lower for keyword in ["compare", "vs", "versus", "tradeoff", "pros", "cons", "better"]):
            tool_decision = "COMPARE"
        elif any(keyword in step_lower for keyword in ["plan", "roadmap", "recommend", "advice", "should i", "learning", "learn", "study", "realistic plan", "focus on"]):
            tool_decision = "ADVISE"
        elif any(keyword in step_lower for keyword in ["explain", "summarize", "analyze", "understand", "why", "how", "architecture", "roles", "failure", "evaluate"]):
            tool_decision = "REASON"
        elif any(keyword in step_lower for keyword in ["pdf", "document", "paper", "research", "knowledge base", "context", "retrieve", "citation"]):
            tool_decision = "RAG"

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
                "output": llm.invoke(
                    REASONING_PROMPT.format(
                        query=f"Compare these options clearly and practically: {step}"
                    )
                ).content
            }

        elif tool_decision == "ADVISE":
            step_result = {
                "agent": "advisor",
                "status": "success",
                "tool_used": "none",
                "output": llm.invoke(
                    REASONING_PROMPT.format(
                        query=f"Give practical advice and a clear recommendation: {step}"
                    )
                ).content
            }

        elif tool_decision == "RAG":
    # temporary retrieval docs
            retrieved_docs = [
        "RAG stands for Retrieval-Augmented Generation.",
        "A strong RAG pipeline includes chunking, embedding, retrieval, and grounded answer generation.",
        "PDF assistants often use OCR, chunking, vector stores, and citation-aware answering."
    ]

            step_result = rag_agent(step, retrieved_docs)

        else:  # REASON
            step_result = {
                "agent": "reasoner",
                "status": "success",
                "tool_used": "none",
                "output": llm.invoke(
                    REASONING_PROMPT.format(query=step)
                ).content
            }

        results.append(f"{step} -> {step_result['output']}")

    results_text = "\n\n".join(results)

    # Internal reasoning grounded to the actual user task
    thought = llm.invoke(f"""
You are an expert AI systems architect.

USER TASK:
{query}

STEP RESULTS:
{results_text}

Think deeply step-by-step about:
- what the task is REALLY asking
- what concrete answer structure is needed
- realistic architecture / reasoning / tradeoffs
- likely failure points
- useful evaluation criteria

IMPORTANT:
- Stay strictly aligned to the USER TASK
- Do NOT drift into another domain
- Do NOT answer yet
- Just reason carefully
""")

    # Final answer
    final = llm.invoke(f"""
You are generating the final answer for this exact task.

USER TASK:
{query}

STEP RESULTS:
{results_text}

INTERNAL REASONING:
{thought.content}

IMPORTANT:
- Stay strictly aligned to the USER TASK
- Do NOT drift into another domain
- Do NOT give a generic unrelated architecture answer
- Use the actual step results
- Be specific, practical, and structured

Now generate the final answer.
""")

    return {
        "result": final.content,
        "input": query,
        "step_results": results
    }