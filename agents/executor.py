from config import llm, ft_llm
from prompts import TOOL_DECISION_PROMPT, REASONING_PROMPT
from utils.parser import parse_steps
from utils.logger import log_event
from agents.researcher import researcher
from agents.coder import coder
from observability import observe, langfuse
from tools.rag_tool import rag_search
import time


@observe(name="executor-agent", as_type="agent")
def executor(state):
    query = state["input"]
    plan = state["plan"]

    log_event("[EXECUTOR]", "Executing step-by-step...")

    steps = parse_steps(plan)
    if not steps:
        steps = [query]

    results = []
    rolling_context = ""

    for step in steps:
        log_event("Step", step)

        langfuse.update_current_span(
            metadata={"current_step": step}
        )

        # Decide tool using BASE model only
        tool_decision = llm.invoke(
            TOOL_DECISION_PROMPT.format(step=step)
        ).content.strip().upper()

        step_lower = step.lower()

        # Hard routing guardrails
        if any(keyword in step_lower for keyword in ["compare", "vs", "versus", "tradeoff", "pros", "cons", "better"]):
            tool_decision = "COMPARE"
        elif any(keyword in step_lower for keyword in ["plan", "roadmap", "recommend", "advice", "should i", "learning", "learn", "study", "realistic plan", "focus on"]):
            tool_decision = "ADVISE"
        elif any(keyword in step_lower for keyword in ["pdf", "document", "paper", "research", "knowledge base", "context", "retrieve", "citation", "docs"]):
            tool_decision = "RAG"
        elif any(keyword in step_lower for keyword in ["write code", "python", "function", "script", "implement", "code"]):
            tool_decision = "CODE"
        elif any(keyword in step_lower for keyword in ["latest", "news", "today", "recent", "search", "web"]):
            tool_decision = "SEARCH"
        else:
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
            compare_res = llm.invoke(
                REASONING_PROMPT.format(
                    query=f"Compare these options clearly and practically: {step}"
                )
            ).content

            step_result = {
                "agent": "comparer",
                "status": "success",
                "tool_used": "none",
                "output": compare_res
            }

        elif tool_decision == "ADVISE":
            advise_res = llm.invoke(
                REASONING_PROMPT.format(
                    query=f"Give practical advice and a clear recommendation: {step}"
                )
            ).content

            step_result = {
                "agent": "advisor",
                "status": "success",
                "tool_used": "none",
                "output": advise_res
            }

        elif tool_decision == "RAG":
            docs = rag_search(step, top_k=3)

            if not docs:
                step_result = {
                    "agent": "rag",
                    "status": "no_results",
                    "tool_used": "rag",
                    "output": "No relevant document context found."
                }
            else:
                context = "\n\n".join([f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(docs)])

                rag_answer = llm.invoke(f"""
You are answering ONLY using retrieved document context.

USER REQUEST:
{step}

RETRIEVED DOCUMENT CONTEXT:
{context}

IMPORTANT RULES:
- Use ONLY the retrieved context
- Do NOT use outside knowledge
- If the answer is not in the context, say that clearly
- Be concise but informative
""").content

                step_result = {
                    "agent": "rag",
                    "status": "success",
                    "tool_used": "rag",
                    "output": rag_answer
                }

        else:  # REASON
            reasoning_res = llm.invoke(f"""
Context: {rolling_context}

Task: {step}

Instruction: Provide a professional, technically accurate response.
""").content

            step_result = {
                "agent": "reasoner",
                "status": "success",
                "tool_used": "none",
                "output": reasoning_res
            }

        results.append(f"{step} -> {step_result['output']}")
        rolling_context += f"\n{step_result['output']}"

    results_text = "\n\n".join(results)

    # Final answer prompt (STRICT)
    final_prompt = f"""
You are generating the final answer for this exact task.

USER TASK:
{query}

STEP RESULTS:
{results_text}

IMPORTANT RULES:
- Use ONLY the information contained in STEP RESULTS
- Do NOT add outside knowledge
- Do NOT invent examples
- If an example is needed, it MUST come directly from the retrieved document content
- If the retrieved content does not support something, say that clearly
- Keep the answer grounded, specific, and structured

Now generate the final answer.
"""

    # Fine-tuned model ONLY here, with fallback
    try:
        final = ft_llm.invoke(final_prompt)
    except Exception as e:
        print(f"\n⚠️ Fine-tuned model failed, falling back to base model: {e}\n")
        time.sleep(2)
        final = llm.invoke(final_prompt)

    return {
        "result": final.content,
        "input": query,
        "step_results": results
    }