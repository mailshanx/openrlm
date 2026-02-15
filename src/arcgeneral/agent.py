import logging
import json
import os

from openrouter import OpenRouter

from arcgeneral.config import AgentConfig
from arcgeneral.sandbox import Sandbox
from arcgeneral.tool import PYTHON_TOOL_SCHEMA, execute_tool


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant with access to a stateful Python execution environment.

## python tool

Use the python tool to execute code in a persistent IPython kernel. The environment \
is a Jupyter notebook — variables, imports, and function definitions persist across \
calls. Common libraries are pre-installed (numpy, pandas, matplotlib, scipy, etc.).

The tool returns stdout/stderr output from execution, or an error traceback if \
the code fails. If no output is produced, you'll see a confirmation message.

### Guidelines

- Use code execution to verify your reasoning, perform calculations, analyze data, \
and produce visualizations.
- Define reusable functions when you'll need similar logic again — they persist.
- If code fails, read the traceback, fix the issue, and retry.
- For large outputs, summarize rather than dumping raw data.
- Files saved to `/app/downloads/` are accessible on the host machine's ~/Downloads folder.\
"""


async def _run_turn(client: OpenRouter, messages: list, sandbox: Sandbox, config: AgentConfig, request_kwargs: dict) -> str:
    """Run one turn of the agent loop (LLM calls + tool calls until stop). Returns the final text response."""
    for round_num in range(config.max_tool_rounds):
        logger.info("Round %d", round_num + 1)

        response = await client.chat.send_async(
            messages=messages,
            **request_kwargs,
        )

        choice = response.choices[0]
        msg = choice.message

        if msg.content:
            print(f"[LLM] {msg.content}")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                raw_args = tc.function.arguments or "{}"
                args_pretty = json.loads(raw_args).get("code", raw_args)
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

        if str(choice.finish_reason) == "stop" or not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            result = await execute_tool(
                sandbox,
                tc.function.name,
                tc.function.arguments or "{}",
                timeout=config.code_timeout,
            )
            print(f"[tool result] {result}")
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
    if config.host_functions:
        section = config.host_functions.prompt_section()
        if section:
            system_prompt = system_prompt + "\n\n" + section
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
