import asyncio
import logging
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from openrouter import OpenRouter
import httpx

from arcgeneral.config import AgentConfig
from arcgeneral.host_functions import HostFunctionRegistry, HostFunctionServer
from arcgeneral.sandbox import Sandbox
from arcgeneral.tool import PYTHON_TOOL_SCHEMA, execute_tool


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant with access to a stateful Python execution environment.

## Code Execution Environment

You have access to a Python environment via the python tool. This is your ONLY tool.

Available functions inside the Python environment (call as global async functions, no import needed):

{functions_json}

You work in a loop. Each iteration:
1. You reason about the output from the previous step and explain what you will do next
2. You write at most ONE python tool call
3. The system runs it and returns the console output

This loop repeats \u2014 you will get multiple iterations. Do not try to do everything in a single \
python tool call. Break your work into steps: first explore and verify, then build on what works.

When you have the final answer, respond to the user in plain text without calling the python tool.

### Rules
1. The python tool is your ONLY tool. All functions above are called with `await` INSIDE python \
code \u2014 never as separate tool calls.
2. Variables, imports, and function definitions persist across python tool calls.
3. If code fails, read the traceback, fix the issue, and retry.
4. For large outputs, summarize rather than dumping raw data.
5. Use `asyncio.gather()` to parallelize independent searches and fetches within a single step.
6. Files saved to `/app/downloads/` are accessible on the host machine's ~/Downloads folder.
7. To save context space, only your latest user message and its tool interactions are shown in full — earlier exchanges are condensed to user message + final response. Your complete history including all tool calls, outputs, and errors is available as `_conversation_history` — a list of message dicts (role, content, and optionally tool_calls or tool_call_id), updated after each step.

### Scaling with Sub-agents
Each code execution returns at most 2000 lines of output and times out after 7 minutes. \
To handle larger workloads, delegate to sub-agents:
path shown in the truncation message). To handle larger workloads, delegate to sub-agents:
- `agent_id = await create_agent(instructions='...')` — creates an agent with its own Python environment
- `result = await run_agent(agent_id=agent_id, task='...')` — dispatches a task, returns the final response
- Run multiple tasks in parallel: `await asyncio.gather(run_agent(...), run_agent(...))`
- Sub-agents share `/app/downloads/` — use files to pass data between agents.
- The `instructions` parameter is appended to the agent's system prompt — use it for role, \
expertise, and constraints. Use `task` for the actual work.
- Principle of Monotonicity: a sub-agent's task MUST be strictly simpler than your own — delegate proper subtasks, never your entire goal.
"""


@dataclass
class _SubAgent:
    """State for a sub-agent created by the main agent."""
    sandbox: Sandbox
    messages: list
    request_kwargs: dict
    depth: int


class AgentRuntime:
    """Owns the session lifecycle: host function server, sandboxes, sub-agents, and the LLM loop."""

    def __init__(self, config: AgentConfig, registry: HostFunctionRegistry):
        self._config = config
        self._registry = registry
        self._sub_agents: dict[str, _SubAgent] = {}
        self._inflight_tasks: set[asyncio.Task] = set()
        self._http_client: httpx.AsyncClient | None = None
        self._server: HostFunctionServer | None = None
        self._main_sandbox: Sandbox | None = None
        self._client: OpenRouter | None = None
        self._messages: list | None = None
        self._request_kwargs: dict | None = None

        # Resolve host downloads dir from sandbox_binds for output spooling
        self._host_downloads_dir: Path | None = None
        for host_path, container_name in config.sandbox_binds.items():
            if container_name == "downloads":
                self._host_downloads_dir = Path(host_path)
                break

        # Register runtime host functions into the registry (before server starts,
        # so they appear in the system prompt and get kernel stubs).
        registry.register(
            "create_agent",
            self._host_create_agent,
            description=(
                "Create a sub-agent with its own Python environment. "
                "The sub-agent has the same capabilities and functions as you, "
                "plus any additional instructions you provide. "
                "Returns an agent_id string to use with run_agent."
            ),
        )
        registry.register(
            "run_agent",
            self._host_run_agent,
            description=(
                "Run a task on a previously created sub-agent. "
                "The sub-agent executes in its own environment with its own state. "
                "Conversation history persists across calls to the same agent_id. "
                "Returns the sub-agent's final text response."
            ),
        )

    async def __aenter__(self) -> Self:
        # Start shared host function server
        if self._registry.names:
            self._server = HostFunctionServer(self._registry)
            await self._server.__aenter__()

        # Start main sandbox
        self._main_sandbox = Sandbox(
            tag=self._config.sandbox_image,
            binds=self._config.sandbox_binds,
            host_function_server=self._server,
        )
        await self._main_sandbox.__aenter__()

        # Inject agent identity for depth tracking
        await self._inject_agent_id(self._main_sandbox, "main")

        # Init shared LLM client (HTTP/2) and messages
        api_key = os.environ.get(self._config.api_key_env_var)
        if api_key is None:
            raise ValueError(f"{self._config.api_key_env_var} is not set")
        self._http_client = httpx.AsyncClient(http2=True, follow_redirects=True)
        self._client = OpenRouter(api_key=api_key, async_client=self._http_client)
        self._messages, self._request_kwargs = self._init_messages(self._config)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cancel in-flight host function tasks before killing sandboxes
        for t in list(self._inflight_tasks):
            t.cancel()
        if self._inflight_tasks:
            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
        self._inflight_tasks.clear()

        # Kill all sub-agent sandboxes
        for agent_id, sub in list(self._sub_agents.items()):
            try:
                await sub.sandbox.__aexit__(None, None, None)
            except Exception:
                logger.warning("Failed to kill sub-agent %s sandbox", agent_id, exc_info=True)
        self._sub_agents.clear()

        # Kill main sandbox
        if self._main_sandbox is not None:
            try:
                await self._main_sandbox.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                logger.warning("Failed to kill main sandbox", exc_info=True)
            finally:
                self._main_sandbox = None

        # Stop host function server
        if self._server is not None:
            try:
                await self._server.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                logger.warning("Failed to stop HostFunctionServer", exc_info=True)
            finally:
                self._server = None

        # Close shared HTTP client
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                logger.warning("Failed to close HTTP client", exc_info=True)
            finally:
                self._http_client = None
                self._client = None

    def _init_messages(self, config: AgentConfig, extra_instructions: str | None = None) -> tuple[list, dict]:
        """Seed the messages list and build request kwargs."""
        messages: list = []
        system_prompt = config.system_prompt if config.system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        functions_json = self._registry.build_schemas_json()
        system_prompt = system_prompt.format(functions_json=functions_json)
        if extra_instructions:
            system_prompt = system_prompt + "\n\n## Additional Instructions\n" + extra_instructions
        messages.append({"role": "system", "content": system_prompt})

        request_kwargs: dict = {
            "model": config.model,
            "tools": [PYTHON_TOOL_SCHEMA],
        }
        if config.temperature is not None:
            request_kwargs["temperature"] = config.temperature
        return messages, request_kwargs

    async def _inject_agent_id(self, sandbox: Sandbox, agent_id: str) -> None:
        """Inject _AGENT_ID and rewrap create_agent to pass it automatically for depth tracking."""
        code = (
            f"_AGENT_ID = {agent_id!r}\n"
            "if 'create_agent' in dir():\n"
            "    _original_create_agent = create_agent\n"
            "    async def create_agent(instructions):\n"
            "        return await _original_create_agent(instructions=instructions, _caller_id=_AGENT_ID)\n"
        )
        try:
            await sandbox.execute(code, timeout=10.0)
        except Exception:
            logger.debug("Failed to inject agent ID %s", agent_id, exc_info=True)

    async def _sync_history(self, sandbox: Sandbox, messages: list) -> None:
        """Push the full conversation history into the kernel as _conversation_history."""
        try:
            payload = json.dumps(messages, ensure_ascii=False)
            code = f"import json as _json\n_conversation_history = _json.loads({payload!r})"
            await sandbox.execute(code, timeout=10.0)
        except Exception:
            logger.debug("Failed to sync conversation history to kernel", exc_info=True)

    @staticmethod
    def _compress_messages(full_history: list) -> list:
        """Compress previous turns to user+final-assistant only. Current turn kept in full."""
        # Find the last user message — everything from there is the current turn
        last_user_idx = None
        for i in range(len(full_history) - 1, -1, -1):
            if full_history[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return list(full_history)

        # Previous turns: keep system, user, and final assistant (no tool_calls) only
        compressed = []
        for msg in full_history[:last_user_idx]:
            role = msg.get("role")
            if role in ("system", "user"):
                compressed.append(msg)
            elif role == "assistant" and "tool_calls" not in msg:
                compressed.append(msg)
            # Drop: assistant with tool_calls, tool results

        # Current turn: everything from last user message onward
        compressed.extend(full_history[last_user_idx:])
        return compressed

    async def _llm_call(self, messages: list, request_kwargs: dict):
        """LLM call with retry on transient errors (4 attempts, exponential backoff)."""
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                return await self._client.chat.send_async(
                    messages=messages,
                    **request_kwargs,
                )
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                wait = min(2 ** attempt, 10)
                logger.warning("LLM call failed (attempt %d/%d), retrying in %ds: %s", attempt + 1, max_attempts, wait, e)
                await asyncio.sleep(wait)

    async def _run_turn(self, full_history: list, sandbox: Sandbox, config: AgentConfig, request_kwargs: dict, agent_label: str = "main") -> str:
        """Run one turn of the agent loop (LLM calls + tool calls until stop). Returns the final text response."""
        for round_num in range(config.max_tool_rounds):
            compressed = self._compress_messages(full_history)
            response = await self._llm_call(compressed, request_kwargs)

            choice = response.choices[0]
            msg = choice.message
            u = response.usage
            logger.info("[%s] [model=%s finish=%s tokens=%s]", agent_label, response.model, choice.finish_reason, f"{u.prompt_tokens:.0f}+{u.completion_tokens:.0f}" if u else "?")
            if msg.content:
                logger.info("[%s] [LLM] %s", agent_label, msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    raw_args = tc.function.arguments or "{}"
                    try:
                        args_pretty = json.loads(raw_args).get("code", raw_args)
                    except (json.JSONDecodeError, AttributeError):
                        args_pretty = raw_args
                    logger.info("[%s] [tool call] %s:\n%s", agent_label, tc.function.name, args_pretty)

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
            full_history.append(assistant_msg)

            if not msg.tool_calls:
                await self._sync_history(sandbox, full_history)
                return msg.content or ""

            for tc in msg.tool_calls:
                result = await execute_tool(
                    sandbox,
                    tc.function.name,
                    tc.function.arguments or "{}",
                    timeout=config.code_timeout,
                    limit_lines=config.output_limit_lines,
                    limit_bytes=config.output_limit_bytes,
                    host_downloads_dir=self._host_downloads_dir,
                )
                logger.info("[%s] [tool result] %s", agent_label, result)
                full_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            await self._sync_history(sandbox, full_history)

        return full_history[-1].get("content", "") if isinstance(full_history[-1], dict) else ""

    # ── Host functions (called from kernel via HTTP bridge) ──

    async def _host_create_agent(self, instructions: str, _caller_id: str = "main") -> str:
        """Create a sub-agent. Returns agent_id."""
        # Look up calling agent's depth
        caller_sub = self._sub_agents.get(_caller_id)
        depth = caller_sub.depth if caller_sub else 0
        if depth >= self._config.max_sub_agent_depth:
            raise RuntimeError(f"Max sub-agent depth ({self._config.max_sub_agent_depth}) exceeded")

        agent_id = uuid.uuid4().hex[:12]
        logger.info("[runtime] Creating sub-agent %s (caller=%s, depth=%d)", agent_id, _caller_id, depth + 1)

        sandbox = Sandbox(
            tag=self._config.sandbox_image,
            binds=self._config.sandbox_binds,
            host_function_server=self._server,
        )
        await sandbox.__aenter__()
        await self._inject_agent_id(sandbox, agent_id)

        messages, request_kwargs = self._init_messages(
            self._config,
            extra_instructions=instructions,
        )
        self._sub_agents[agent_id] = _SubAgent(
            sandbox=sandbox,
            messages=messages,
            request_kwargs=request_kwargs,
            depth=depth + 1,
        )

        logger.info("[runtime] Sub-agent %s ready (depth=%d)", agent_id, depth + 1)
        return agent_id

    async def _host_run_agent(self, agent_id: str, task: str) -> str:
        """Run a task on a sub-agent. Returns the final text response."""
        current = asyncio.current_task()
        if current is not None:
            self._inflight_tasks.add(current)
        try:
            sub = self._sub_agents.get(agent_id)
            if sub is None:
                raise RuntimeError(f"Unknown agent_id: {agent_id}")

            logger.info("[runtime] Running task on sub-agent %s: %s", agent_id, task[:100])
            sub.messages.append({"role": "user", "content": task})
            result = await self._run_turn(sub.messages, sub.sandbox, self._config, sub.request_kwargs, agent_label=agent_id)
            logger.info("[runtime] Sub-agent %s finished", agent_id)
            return result
        finally:
            if current is not None:
                self._inflight_tasks.discard(current)

    # ── Public API ──

    async def run_single(self, user_message: str) -> str:
        """Run the agent loop for a single message. Returns the final text response."""
        logger.info("[user] %s", user_message)
        self._messages.append({"role": "user", "content": user_message})
        result = await self._run_turn(self._messages, self._main_sandbox, self._config, self._request_kwargs, agent_label="main")
        logger.info("[assistant] %s", result)
        return result

    async def run_session(self) -> None:
        """Interactive session: accept user messages in a loop, run agent turns, print responses."""
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

            logger.info("[user] %s", stripped)
            self._messages.append({"role": "user", "content": stripped})
            result = await self._run_turn(self._messages, self._main_sandbox, self._config, self._request_kwargs, agent_label="main")
            logger.info("[assistant] %s", result)
            print(result)
            print()
