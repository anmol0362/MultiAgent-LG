from config import llm
from prompts import CODE_PROMPT
from tools.python_tool import python_tool
from utils.logger import log_event
from observability import observe

@observe(name="coder-agent", as_type="agent")
def coder(query: str):
    log_event("[CODER AGENT]", f"Task: {query}")

    code = llm.invoke(
        CODE_PROMPT.format(query=query)
    ).content.strip()

    log_event("Generated Code", code)

    try:
        output = python_tool.invoke(code)
        status = "success"
    except Exception as e:
        output = str(e)
        status = "error"

    return {
        "agent": "coder",
        "status": status,
        "tool_used": "python",
        "output": f"Code:\n{code}\n\nOutput:\n{output}"
    }