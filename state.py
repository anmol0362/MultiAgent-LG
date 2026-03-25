from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    input: str
    plan: List[str]
    current_step: Optional[str]
    step_results: List[Dict[str, Any]]
    result: str
    route: str
    errors: List[str]