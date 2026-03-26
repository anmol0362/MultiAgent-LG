from typing import TypedDict, List, Annotated

class AgentState(TypedDict):
    input: str
    plan: str
    step_results: List[str]
    result: str
    quality_score: int  # Add this
    feedback: str       # Add this