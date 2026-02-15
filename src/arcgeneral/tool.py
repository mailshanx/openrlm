import json

from arcgeneral.sandbox import Sandbox


PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python code in a stateful IPython environment. "
            "Variables, imports, and function definitions persist across calls."
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
    args = json.loads(arguments)
    return await sandbox.execute(args["code"], timeout=timeout)
