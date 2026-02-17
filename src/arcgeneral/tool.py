import json
import os
import uuid
from pathlib import Path

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


def _truncate_and_spool(
    output: str,
    *,
    limit_lines: int,
    limit_bytes: int,
    host_spool_dir: Path,
    container_spool_dir: str,
) -> str:
    """If output exceeds limits, spool full output to disk and return truncated version."""
    lines = output.split("\n")
    over_lines = len(lines) > limit_lines
    over_bytes = len(output.encode("utf-8", errors="replace")) > limit_bytes

    if not over_lines and not over_bytes:
        return output

    # Truncate to whichever limit hits first
    truncated_lines = lines[:limit_lines]
    truncated = "\n".join(truncated_lines)
    if len(truncated.encode("utf-8", errors="replace")) > limit_bytes:
        encoded = truncated.encode("utf-8", errors="replace")[:limit_bytes]
        truncated = encoded.decode("utf-8", errors="ignore")

    # Spool full output
    host_spool_dir.mkdir(parents=True, exist_ok=True)
    filename = f"output_{uuid.uuid4().hex[:12]}.log"
    host_path = host_spool_dir / filename
    host_path.write_text(output, encoding="utf-8")

    container_path = f"{container_spool_dir}/{filename}"

    return (
        f"{truncated}\n\n"
        f"[OUTPUT TRUNCATED — {len(lines)} lines / {len(output.encode('utf-8', errors='replace'))} bytes exceeded limit. "
        f"Full output saved to {container_path} — read it with: open('{container_path}').read()]"
    )


async def execute_tool(
    sandbox: Sandbox,
    name: str,
    arguments: str,
    timeout: float,
    limit_lines: int = 2000,
    limit_bytes: int = 50_000,
    host_workspace_dir: Path | None = None,
) -> str:
    """Dispatch a tool call. Returns the result string, truncated if needed."""
    if name != "python":
        return f"Unknown tool: {name}"
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return f"Error: invalid tool call arguments: {arguments!r}"
    code = args.get("code")
    if not code:
        return "Error: no code provided in tool call arguments."
    result = await sandbox.execute(code, timeout=timeout)

    if host_workspace_dir is not None:
        result = _truncate_and_spool(
            result,
            limit_lines=limit_lines,
            limit_bytes=limit_bytes,
            host_spool_dir=host_workspace_dir / ".arcgeneral_spool",
            container_spool_dir="/app/workspace/.arcgeneral_spool",
        )

    return result
