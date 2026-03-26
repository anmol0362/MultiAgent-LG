from config import ft_llm, llm
from observability import observe

@observe(name="reviewer-agent", as_type="agent")
def reviewer(state):
    query = state["input"]
    final_answer = state["result"]
    
    # Use the Fine-Tuned model to judge the quality
    review_prompt = f"""
    You are an NTSB Safety Inspector reviewing an AI-generated report.
    
    ORIGINAL QUERY: {query}
    AI RESPONSE: {final_answer}
    
    CRITIQUE THE RESPONSE BASED ON:
    1. Technical Accuracy (Does it align with FAA/Aviation standards?)
    2. Completeness (Did it answer all parts of the user query?)
    3. Grounding (Does it sound like it's based on real incident data or just generic AI talk?)

    OUTPUT FORMAT:
    SCORE: [0-10]
    FEEDBACK: [One sentence explaining why it got this score and what is missing]
    """
    
    response = ft_llm.invoke(review_prompt).content
    
    # Simple parsing logic
    try:
        score_line = [l for l in response.split('\n') if "SCORE:" in l][0]
        score = int(''.join(filter(str.isdigit, score_line)))
        feedback = [l for l in response.split('\n') if "FEEDBACK:" in l][0]
    except:
        score = 5
        feedback = "Reviewer failed to parse. Re-evaluating."

    print(f"\n⭐ Reviewer Score: {score}/10")
    print(f"📝 Feedback: {feedback}\n")

    return {
        "quality_score": score,
        "feedback": feedback
    }