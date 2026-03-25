from config import llm
from prompts import RESEARCH_ANSWER_PROMPT
from tools.search_tool import search_tool
from utils.logger import log_event
from observability import observe

@observe(name="researcher-agent", as_type="agent")
def researcher(query: str, use_search: bool = True):
    log_event("[RESEARCHER AGENT]", f"Query: {query}")

    if use_search:
        log_event("Search", "Using Tavily search...")
        tool_data = search_tool.invoke(query)
        tool_used = "search"
    else:
        log_event("Search", "Skipping search, using reasoning only.")
        tool_data = "No external search used."
        tool_used = "none"

    final = llm.invoke(
        RESEARCH_ANSWER_PROMPT.format(query=query, tool_data=tool_data)
    )

    return {
        "agent": "researcher",
        "status": "success",
        "tool_used": tool_used,
        "output": final.content
    }