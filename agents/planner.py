from config import llm
from prompts import PLANNER_PROMPT
from observability import observe

@observe(name="planner-agent", as_type="agent")
def planner(state):
    query = state["input"]
    feedback = state.get("feedback", "No previous feedback.")
    previous_steps = state.get("step_results", [])

    # If this is the second attempt, we tell the LLM exactly what went wrong
    if feedback != "No previous feedback.":
        system_instructions = f"""
        You are an expert Aviation Project Planner.
        
        CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT:
        {feedback}
        
        PREVIOUS ATTEMPT STEPS:
        {previous_steps}
        
        The previous plan failed to meet safety standards. 
        Create a NEW, more detailed plan that addresses the feedback specifically.
        Focus on retrieving technical data (RAG) or performing deeper analysis.
        """
    else:
        system_instructions = "You are an expert Aviation Project Planner. Break the user query into logical steps."

    prompt = f"""
    {system_instructions}
    
    USER TASK: {query}
    
    Format the plan as a numbered list of steps. 
    Example:
    1. Retrieve ASRS data on winglet alterations.
    2. Analyze FAA regulatory compliance for avionics.
    """

    response = llm.invoke(prompt).content
    print(f"\n🗺️ New Plan Generated:\n{response}")

    return {"plan": response}