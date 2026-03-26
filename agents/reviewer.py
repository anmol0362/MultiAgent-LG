from config import llm

def reviewer(state):
    task = state.get("task", "")
    result = state.get("result", "")

    prompt = f"""
You are a strict evaluator for a multi-agent AI system.

Task:
{task}

Result:
{result}

Give a short evaluation on:
1. planning quality
2. reasoning depth
3. usefulness
4. whether it feels generic or strong

Return in this format exactly:

PLANNER_SCORE: <0-10>
FINAL_SCORE: <0-10>
FEEDBACK: <one short sentence>
"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    state["review"] = content
    return state