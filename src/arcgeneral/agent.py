import logging
import json
import os
from pathlib import Path
import sys

from openrouter import OpenRouter

from arcgeneral.config import AgentConfig
from arcgeneral.sandbox import Sandbox
from arcgeneral.tool import PYTHON_TOOL_SCHEMA, execute_tool


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant with access to a stateful Python execution environment.
## Code Execution Environment

You have access to a Python sandbox via the python tool. This is your ONLY tool.

Available functions inside the Python sandbox (call as global async functions, no import needed):

{functions_json}

You work in a loop. Each iteration:
1. You reason about the output from the previous step and explain what you will do next
2. You write at most ONE python tool call
3. The system runs it and returns the console output

This loop repeats — you will get multiple iterations. Do not try to do everything in a single \
python tool call. Break your work into steps: first explore and verify, then build on what works.

When you have the final answer, respond to the user in plain text without calling the python tool.

### Rules
1. The python tool is your ONLY tool. All functions above are called with `await` INSIDE python \
code — never as separate tool calls.
2. Variables, imports, and function definitions persist across python tool calls.
3. If code fails, read the traceback, fix the issue, and retry.
4. For large outputs, summarize rather than dumping raw data.
5. Use `asyncio.gather()` to parallelize independent searches and fetches within a single step.
6. Files saved to `/app/downloads/` are accessible on the host machine's ~/Downloads folder.\
"""


async def _run_turn(client: OpenRouter, messages: list, sandbox: Sandbox, config: AgentConfig, request_kwargs: dict) -> str:
    """Run one turn of the agent loop (LLM calls + tool calls until stop). Returns the final text response."""
    # Resolve host downloads dir from sandbox_binds for output spooling
    host_downloads_dir = None
    for host_path, container_name in config.sandbox_binds.items():
        if container_name == "downloads":
            host_downloads_dir = Path(host_path)
            break
    for round_num in range(config.max_tool_rounds):

        response = await client.chat.send_async(
            messages=messages,
            **request_kwargs,
        )

        choice = response.choices[0]
        msg = choice.message
        u = response.usage
        print(f"[model={response.model} finish={choice.finish_reason} tokens={f'{u.prompt_tokens:.0f}+{u.completion_tokens:.0f}' if u else '?'}]")
        if msg.content:
            print(f"[LLM] {msg.content}")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                raw_args = tc.function.arguments or "{}"
                try:
                    args_pretty = json.loads(raw_args).get("code", raw_args)
                except (json.JSONDecodeError, AttributeError):
                    args_pretty = raw_args
                print(f"[tool call] {tc.function.name}:\n{args_pretty}")

        # Convert to dict for round-tripping back into messages
        assistant_msg = {"role": "assistant"}
        assistant_msg["content"] = msg.content or None
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            result = await execute_tool(
                sandbox,
                tc.function.name,
                tc.function.arguments or "{}",
                timeout=config.code_timeout,
                limit_lines=config.output_limit_lines,
                limit_bytes=config.output_limit_bytes,
                host_downloads_dir=host_downloads_dir,
            )
            preview = result[:200] + '...' if len(result) > 1000 else result
            print(f"[tool result] {preview}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""


def _init_client_and_messages(config: AgentConfig) -> tuple[OpenRouter, list, dict]:
    """Create the OpenRouter client, seed the messages list, and build request kwargs."""
    api_key = os.environ.get(config.api_key_env_var)
    if api_key is None:
        raise ValueError(f"{config.api_key_env_var} is not set")

    client = OpenRouter(api_key=api_key)

    messages: list = []
    system_prompt = config.system_prompt if config.system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    functions_json = config.host_functions.build_schemas_json() if config.host_functions else "(none)"
    system_prompt = system_prompt.format(functions_json=functions_json)
    messages.append({"role": "system", "content": system_prompt})

    request_kwargs: dict = {
        "model": config.model,
        "tools": [PYTHON_TOOL_SCHEMA],
    }
    if config.temperature is not None:
        request_kwargs["temperature"] = config.temperature

    return client, messages, request_kwargs


async def run_agent(config: AgentConfig, user_message: str) -> str:
    """Run the agent loop for a single message. Returns the final text response."""
    client, messages, request_kwargs = _init_client_and_messages(config)
    messages.append({"role": "user", "content": user_message})

    async with Sandbox(tag=config.sandbox_image, binds=config.sandbox_binds, host_functions=config.host_functions) as sandbox:
        return await _run_turn(client, messages, sandbox, config, request_kwargs)


async def run_session(config: AgentConfig) -> None:
    """Interactive session: accept user messages in a loop, run agent turns, print responses."""
    client, messages, request_kwargs = _init_client_and_messages(config)

    async with Sandbox(tag=config.sandbox_image, binds=config.sandbox_binds, host_functions=config.host_functions) as sandbox:
        print("arcgeneral session started. Type 'quit' or 'exit' to end.\n")
        while True:
            try:
                user_input = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            stripped = user_input.strip()
            if stripped.lower() in ("quit", "exit"):
                break
            if not stripped:
                continue

            messages.append({"role": "user", "content": stripped})
            result = await _run_turn(client, messages, sandbox, config, request_kwargs)
            print()
