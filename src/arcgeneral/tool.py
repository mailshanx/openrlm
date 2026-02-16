import json

from arcgeneral.sandbox import Sandbox


PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python code in a persistent IPython environment. "
            "Variables, imports, and function definitions persist across calls. "
            "Common libraries are pre-installed (numpy, pandas, matplotlib, scipy, etc.). "
            "Returns stdout/stderr output, or an error traceback if the code fails."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


async def execute_tool(sandbox: Sandbox, name: str, arguments: str, timeout: float) -> str:
    """Dispatch a tool call. Returns the result string."""
    if name != "python":
        return f"Unknown tool: {name}"
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return f"Error: invalid tool call arguments: {arguments!r}"
    code = args.get("code")
    if not code:
        return "Error: no code provided in tool call arguments."
    return await sandbox.execute(code, timeout=timeout)
