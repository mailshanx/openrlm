import asyncio
import logging
import json
import os
import time
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from arcgeneral.llm import LLMClient, OpenRouterClient, default_api_key_resolver

from arcgeneral.config import AgentConfig
from arcgeneral.host_functions import HostFunctionRegistry, HostFunctionServer
from arcgeneral.sandbox import ForkServer, LocalForkServer, Sandbox
from arcgeneral.tool import PYTHON_TOOL_SCHEMA, execute_tool
from arcgeneral.events import AgentEvent, RoundStart, ModelRequest, ModelResponse, ToolExecStart, ToolExecEnd, TurnEnd


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant with access to a stateful Python execution environment.

## Code Execution Environment

You have access to a Python REPL environment via the python tool. This is your ONLY tool.

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
1. The python tool is your ONLY tool. It executes code in a persistent Python REPL. All functions above are called with `await` INSIDE python \
code \u2014 never as separate tool calls.
2. If code fails, read the traceback, fix the issue, and retry.
3. Use `asyncio.gather()` to run independent tasks concurrently within a single python tool call.
4. The working directory `{workspace_path}` is shared with the host. Files you create or modify there are visible on the host, and vice versa.
5. If a package is not installed, run `subprocess.run(["uv", "pip", "install", "<package>"])` to install it.
6. To save context space, only your latest user message and its tool interactions are shown in full \
\u2014 earlier exchanges are condensed to user message + final response. Your complete history \
including all tool calls, outputs, and errors is available as `_conversation_history` \u2014 a list \
of message dicts (role, content, and optionally tool_calls or tool_call_id), updated after each step.

### Python REPL
State persists across calls — variables, imports, and function definitions all survive between tool calls.
Build up results incrementally rather than repeating work or relying on conversation context.
When storing results, retain provenance alongside extracted facts.
Print only what changed or a brief status, not raw output.
  # first call
  data = [row for row in csv.reader(open(f'{workspace_path}/input.csv'))]
  summary = {{"rows": len(data), "columns": len(data[0])}}
  print(f"Loaded {{summary['rows']}} rows")

  # later call — all state from prior calls is still here
  filtered = [r for r in data if float(r[2]) > threshold]
  print(f"Filtered to {{len(filtered)}} rows")

For data too large to hold in variables, or that other agents need, write to `{workspace_path}`.
Read it back in later calls rather than re-doing work.

### Scaling with Sub-agents
You work in a loop, one code block at a time. Each code execution returns at most 2000 lines
of output and times out after 7 minutes. Sub-agents break through these limits — each runs in
its own environment with its own limits, working in the background while you continue:

- agent_id = await create_agent(instructions='...') — creates a sub-agent with its own Python environment
- task_id = await run_agent(agent_id=agent_id, task='...') — starts a task in the background, returns immediately
- result = await await_result(task_id) — blocks until the task finishes, returns the final response

run_agent is non-blocking — the sub-agent works in the background while your code continues.
Submit all tasks first, do your own work, then collect results across multiple steps:

  a = await create_agent(instructions='citation specialist')
  b = await create_agent(instructions='judicial historian')
  t1 = await run_agent(agent_id=a, task='research citation percentiles')
  t2 = await run_agent(agent_id=b, task='compile judge case histories')

Then continue your own work — sub-agents are running in the background. When ready:

  r1, r2 = await asyncio.gather(await_result(t1), await_result(t2))

- Sub-agents share `{workspace_path}` — use files to pass large data between agents.
- Principle of Monotonicity: a sub-agent's task MUST be strictly simpler than your own — delegate proper subtasks, never your entire goal.

### Effective delegation
Sub-agents start with no knowledge of your work so far. They see only their `instructions` and `task`.

- **instructions** defines the agent's role. Be specific about what it should know and do.
- **task** must be self-contained. Include what to do, what you already know that's relevant \
(key findings, file paths, API details, constraints), and what output format you need. \
Do not assume the sub-agent will discover or read files you have not mentioned.
- When spawning agents that build on earlier results (yours or other sub-agents'), include \
the relevant findings directly in the task.

### Using results
After await_result(), understand what the sub-agent produced before deciding next steps. \
When delegating follow-up work, carry forward relevant findings from completed work into new tasks.
"""


@dataclass
class _SubAgent:
    """State for a sub-agent created by the main agent."""
    instructions: str
    depth: int
    sandbox: Sandbox | None = None
    messages: list | None = None
    request_kwargs: dict | None = None
    lock: asyncio.Lock = None  # serializes _run_turn calls per sub-agent

    def __post_init__(self):
        self.lock = asyncio.Lock()


class Session:
    """A single conversation with its own REPL, message history, and sub-agent tree."""

    def __init__(self, runtime: 'AgentRuntime', sandbox: Sandbox, messages: list,
                 request_kwargs: dict, session_id: str, on_event=None):
        self._runtime = runtime
        self._sandbox = sandbox
        self._messages = messages
        self._request_kwargs = request_kwargs
        self._session_id = session_id
        self._on_event = on_event
        self._sub_agents: dict[str, _SubAgent] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    @property
    def session_id(self) -> str:
        """The caller-provided session ID."""
        return self._session_id
    async def run_single(self, user_message: str) -> str:
        """Run the agent loop for a single message. Returns the final text response.

        On CancelledError, cancels all sub-agent tasks spawned during this
        turn so the entire agent tree is interrupted, then re-raises.
        """
        async with self._lock:
            logger.info("[user] %s", user_message)
            self._messages.append({"role": "user", "content": user_message})
            try:
                result = await self._runtime._run_turn(
                    self._messages, self._sandbox, self._runtime._config,
                    self._request_kwargs, agent_label="main",
                    on_event=self._on_event,
                )
                logger.info("[assistant] %s", result)
                return result
            except asyncio.CancelledError:
                # Cancel all sub-agent tasks spawned during this turn
                n_tasks = len(self._running_tasks)
                for task_id, t in list(self._running_tasks.items()):
                    t.cancel()
                if self._running_tasks:
                    await asyncio.gather(
                        *self._running_tasks.values(),
                        return_exceptions=True,
                    )
                self._running_tasks.clear()
                logger.info("[session %s] Turn cancelled, %d sub-agent tasks cleaned up",
                            self._session_id, n_tasks)
                raise

    async def close(self):
        """Close this session: cancel tasks, destroy sandboxes, unregister from runtime."""
        # Cancel running background tasks
        for task_id, t in list(self._running_tasks.items()):
            t.cancel()
            self._runtime._task_to_session.pop(task_id, None)
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        self._running_tasks.clear()

        # Destroy all sub-agent sandboxes
        for agent_id, sub in list(self._sub_agents.items()):
            if sub.sandbox is not None:
                try:
                    await sub.sandbox.close()
                except Exception:
                    logger.warning("Failed to destroy sub-agent %s sandbox", agent_id, exc_info=True)
            self._runtime._agent_to_session.pop(agent_id, None)
        self._sub_agents.clear()

        # Destroy session sandbox
        if self._sandbox is not None:
            try:
                await self._sandbox.close()
            except Exception:
                logger.warning("Failed to destroy session sandbox", exc_info=True)
            finally:
                self._sandbox = None

        # Unregister from runtime
        self._runtime._agent_to_session.pop(self._session_id, None)
        self._runtime._sessions.pop(self._session_id, None)


class AgentRuntime:
    """Owns the infrastructure lifecycle: host function server, fork server, LLM client.
    Creates sessions for independent conversations."""

    def __init__(self, config: AgentConfig, registry: HostFunctionRegistry, llm_client: LLMClient | None = None):
        self._config = config
        self._registry = registry
        self._server: HostFunctionServer | None = None
        self._fork_server: ForkServer | None = None
        self._llm_client = llm_client
        self._owns_llm_client = llm_client is None

        # Session tracking
        self._sessions: dict[str, Session] = {}
        self._agent_to_session: dict[str, Session] = {}
        self._task_to_session: dict[str, Session] = {}

        # Spool dir for truncated output (temp dir, cleaned up on __aexit__)
        self._spool_dir: Path | None = None

        # Register runtime host functions into the registry (before server starts,
        # so they appear in the system prompt and get kernel stubs).
        registry.register("create_agent", self._host_create_agent)
        registry.register("run_agent", self._host_run_agent)
        registry.register(
            "await_result",
            self._host_await_result,
            timeout=max(config.code_timeout - 30, 30),
        )

    async def __aenter__(self) -> Self:
        # Start shared host function server
        if self._registry.names:
            self._server = HostFunctionServer(self._registry)
            await self._server.__aenter__()

        # Create temp spool dir
        self._spool_dir = Path(tempfile.mkdtemp(prefix="arcgeneral_spool_"))
        if self._config.sandbox_image:
            # Docker mode: bind-mount workspace and spool into container
            binds = dict(self._config.sandbox_binds)
            binds[str(self._spool_dir)] = "spool"
            self._fork_server = ForkServer(
                tag=self._config.sandbox_image,
                binds=binds,
                host_function_server=self._server,
            )
            # Resolve paths as they appear inside the container
            self._config.workspace_path = self._config.workspace_path or "/app/workspace/"
            self._config.spool_path = self._config.spool_path or "/app/spool"
        else:
            # Local mode: fork server runs as a subprocess on the host
            self._fork_server = LocalForkServer(
                host_function_server=self._server,
            )
            # Resolve paths to real host directories
            workspace_candidates = list(self._config.sandbox_binds.values())
            if workspace_candidates:
                # Use the first bind's host path as workspace
                workspace_host = list(self._config.sandbox_binds.keys())[0]
                self._config.workspace_path = self._config.workspace_path or workspace_host
            else:
                self._config.workspace_path = self._config.workspace_path or str(Path.cwd())
            self._config.spool_path = self._config.spool_path or str(self._spool_dir)
        await self._fork_server.__aenter__()

        # Init API key resolver (default reads from env vars)
        if self._config.get_api_key is None:
            self._config.get_api_key = default_api_key_resolver()
        # Init LLM client (create default if none injected)
        if self._llm_client is None:
            self._llm_client = OpenRouterClient()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close all sessions
        for session in list(self._sessions.values()):
            try:
                await session.close()
            except Exception:
                logger.warning("Failed to close session", exc_info=True)
        self._sessions.clear()
        self._agent_to_session.clear()
        self._task_to_session.clear()

        # Stop fork server (kills the container)
        if self._fork_server is not None:
            try:
                await self._fork_server.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                logger.warning("Failed to stop fork server", exc_info=True)
            finally:
                self._fork_server = None

        # Stop host function server
        if self._server is not None:
            try:
                await self._server.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                logger.warning("Failed to stop HostFunctionServer", exc_info=True)
            finally:
                self._server = None

        # Close LLM client (only if we created it)
        if self._owns_llm_client and self._llm_client is not None:
            try:
                await self._llm_client.close()
            except Exception:
                logger.warning("Failed to close LLM client", exc_info=True)
            finally:
                self._llm_client = None

        # Clean up spool dir
        if self._spool_dir is not None:
            try:
                shutil.rmtree(self._spool_dir, ignore_errors=True)
            finally:
                self._spool_dir = None

    # ── Session management ──

    async def create_session(self, session_id: str, *, on_event=None, context_messages: list[dict] | None = None) -> Session:
        """Create a new independent session with its own sandbox and message history.

        Args:
            session_id: Caller-provided unique identifier for this session.
            on_event: Optional callback for session events.
            context_messages: Optional list of {"role": "user"|"assistant", "content": "..."}
                messages to prepend as conversation history after the system prompt.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id!r} already exists")
        sandbox = await self._fork_server.create_sandbox()
        await self._inject_agent_id(sandbox, session_id)
        messages, request_kwargs = self._init_messages(self._config)
        if context_messages:
            messages.extend(context_messages)
        session = Session(
            runtime=self,
            sandbox=sandbox,
            messages=messages,
            request_kwargs=request_kwargs,
            session_id=session_id,
            on_event=on_event,
        )
        self._agent_to_session[session_id] = session
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        """Retrieve an existing session by ID."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session: {session_id!r}")
        return session

    async def close_session(self, session_id: str) -> None:
        """Close and remove a session by ID. No-op if session doesn't exist."""
        session = self._sessions.get(session_id)
        if session is not None:
            await session.close()

    def _init_messages(self, config: AgentConfig, extra_instructions: str | None = None) -> tuple[list, dict]:
        """Seed the messages list and build request kwargs."""
        messages: list = []
        system_prompt = config.system_prompt if config.system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        functions_json = self._registry.build_schemas_json()
        system_prompt = system_prompt.format(
            functions_json=functions_json,
            workspace_path=config.workspace_path or "/app/workspace/",
            spool_path=config.spool_path or "/app/spool",
        )
        if extra_instructions:
            system_prompt += (
                "\n\n## Your Role\n"
                "You were created by another agent to accomplish a specific task. "
                "Your final text response (when you stop calling tools) is returned to your creator as the result. "
                "Be specific and structured in your output \u2014 it feeds into further work. "
                "For large results, save to disk and return the file path in your response.\n"
                "\n## Additional Instructions\n" + extra_instructions
            )
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

    @staticmethod
    def _emit(on_event, event: AgentEvent) -> None:
        """Fire an event to the consumer callback, silently ignoring errors."""
        if on_event is not None:
            try:
                on_event(event)
            except Exception:
                logger.debug("Event callback error", exc_info=True)


    async def _llm_call(self, messages: list, request_kwargs: dict):
        """LLM call with retry on transient errors (4 attempts, exponential backoff)."""
        max_attempts = 4
        api_key = await self._config.get_api_key(self._config.provider)
        for attempt in range(max_attempts):
            try:
                return await self._llm_client.complete(
                    messages=messages,
                    api_key=api_key,
                    **request_kwargs,
                )
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                wait = min(2 ** attempt, 10)
                logger.warning("LLM call failed (attempt %d/%d), retrying in %ds: %s", attempt + 1, max_attempts, wait, e)
                await asyncio.sleep(wait)
                # Re-resolve key on retry (may have been refreshed)
                api_key = await self._config.get_api_key(self._config.provider)

    async def _run_turn(self, full_history: list, sandbox: Sandbox, config: AgentConfig, request_kwargs: dict, agent_label: str = "main", on_event=None) -> str:
        """Run one turn of the agent loop (LLM calls + tool calls until stop).

        On CancelledError, rolls back full_history to the start of the
        interrupted round so the message list stays consistent (every
        assistant tool_calls entry has matching tool results).
        """
        turn_start = time.monotonic()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for round_num in range(config.max_tool_rounds):
            checkpoint = len(full_history)
            try:
                self._emit(on_event, RoundStart(agent_id=agent_label, round_num=round_num, max_rounds=config.max_tool_rounds))
                compressed = self._compress_messages(full_history)
                self._emit(on_event, ModelRequest(agent_id=agent_label))
                response = await self._llm_call(compressed, request_kwargs)

                choice = response.choices[0]
                msg = choice.message
                u = response.usage
                pt = int(u.prompt_tokens) if u and u.prompt_tokens else 0
                ct = int(u.completion_tokens) if u and u.completion_tokens else 0
                total_prompt_tokens += pt
                total_completion_tokens += ct
                self._emit(on_event, ModelResponse(
                    agent_id=agent_label, model=response.model or "",
                    finish_reason=choice.finish_reason or "",
                    has_tool_calls=bool(msg.tool_calls),
                    prompt_tokens=pt if u else None,
                    completion_tokens=ct if u else None,
                ))
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
                    self._emit(on_event, TurnEnd(
                        agent_id=agent_label, rounds=round_num + 1,
                        elapsed_seconds=time.monotonic() - turn_start,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                    ))
                    return msg.content or ""

                for tc in msg.tool_calls:
                    raw_args_tc = tc.function.arguments or "{}"
                    try:
                        code_text = json.loads(raw_args_tc).get("code", raw_args_tc)
                    except (json.JSONDecodeError, AttributeError):
                        code_text = raw_args_tc
                    self._emit(on_event, ToolExecStart(agent_id=agent_label, tool_name=tc.function.name, code=code_text))
                    tool_start = time.monotonic()
                    result = await execute_tool(
                        sandbox,
                        tc.function.name,
                        tc.function.arguments or "{}",
                        timeout=config.code_timeout,
                        limit_lines=config.output_limit_lines,
                        limit_bytes=config.output_limit_bytes,
                        host_spool_dir=self._spool_dir,
                        container_spool_dir=config.spool_path or "/app/spool",
                    )
                    self._emit(on_event, ToolExecEnd(agent_id=agent_label, tool_name=tc.function.name, elapsed_seconds=time.monotonic() - tool_start))
                    logger.info("[%s] [tool result] %s", agent_label, result)
                    full_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                await self._sync_history(sandbox, full_history)

            except asyncio.CancelledError:
                del full_history[checkpoint:]
                logger.info("[%s] Cancelled during round %d, rolled back history to %d messages", agent_label, round_num, checkpoint)
                raise

        self._emit(on_event, TurnEnd(
            agent_id=agent_label, rounds=config.max_tool_rounds,
            elapsed_seconds=time.monotonic() - turn_start,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        ))
        return full_history[-1].get("content", "") if isinstance(full_history[-1], dict) else ""

    # ── Host functions (called from kernel via HTTP bridge) ──

    async def _host_create_agent(self, instructions: str, _caller_id: str = "main") -> str:
        """Create a sub-agent with its own Python environment.

        The sub-agent has the same capabilities and functions as you,
        plus any additional instructions you provide.
        Returns an agent_id string to use with run_agent."""
        session = self._agent_to_session.get(_caller_id)
        if session is None:
            raise RuntimeError(f"Unknown caller: {_caller_id}")
        # Look up calling agent's depth
        caller_sub = session._sub_agents.get(_caller_id)
        depth = caller_sub.depth if caller_sub else 0
        if depth >= self._config.max_sub_agent_depth:
            raise RuntimeError(f"Max sub-agent depth ({self._config.max_sub_agent_depth}) exceeded")
        agent_id = uuid.uuid4().hex[:12]
        logger.info("[runtime] Registered sub-agent %s (caller=%s, depth=%d)", agent_id, _caller_id, depth + 1)
        session._sub_agents[agent_id] = _SubAgent(
            instructions=instructions,
            depth=depth + 1,
        )
        self._agent_to_session[agent_id] = session
        return agent_id

    async def _ensure_sandbox(self, agent_id: str, sub: _SubAgent) -> None:
        """Lazily create sandbox and init messages on first use."""
        if sub.sandbox is not None:
            return
        logger.info("[runtime] Creating sandbox for sub-agent %s", agent_id)
        sub.sandbox = await self._fork_server.create_sandbox()
        await self._inject_agent_id(sub.sandbox, agent_id)
        sub.messages, sub.request_kwargs = self._init_messages(
            self._config,
            extra_instructions=sub.instructions,
        )
        logger.info("[runtime] Sub-agent %s ready", agent_id)

    async def _host_run_agent(self, agent_id: str, task: str) -> str:
        """Start a task on a previously created sub-agent.

        Returns a task_id immediately — the sub-agent runs in the background.
        Use await_result(task_id) to collect the result."""
        session = self._agent_to_session.get(agent_id)
        if session is None:
            raise RuntimeError(f"Unknown agent_id: {agent_id}")
        sub = session._sub_agents.get(agent_id)
        if sub is None:
            raise RuntimeError(f"Unknown agent_id: {agent_id}")
        task_id = uuid.uuid4().hex[:12]
        logger.info("[runtime] Starting task %s on sub-agent %s: %s", task_id, agent_id, task[:100])

        async def _execute():
            try:
                async with sub.lock:
                    await self._ensure_sandbox(agent_id, sub)
                    sub.messages.append({"role": "user", "content": task})
                    result = await self._run_turn(sub.messages, sub.sandbox, self._config, sub.request_kwargs, agent_label=agent_id, on_event=session._on_event)
                    logger.info("[runtime] Task %s on sub-agent %s finished", task_id, agent_id)
                    return result
            except asyncio.CancelledError:
                logger.info("[runtime] Task %s on sub-agent %s cancelled", task_id, agent_id)
                raise

        task_obj = asyncio.create_task(_execute())
        session._running_tasks[task_id] = task_obj
        self._task_to_session[task_id] = session
        return task_id

    async def _host_await_result(self, task_id: str) -> str:
        """Block until a background task completes and return the result.

        Pass the task_id returned by run_agent."""
        session = self._task_to_session.get(task_id)
        if session is None:
            raise RuntimeError(f"Unknown task_id: {task_id}")
        task = session._running_tasks.get(task_id)
        if task is None:
            raise RuntimeError(f"Unknown task_id: {task_id}")
        try:
            return await task
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            session._running_tasks.pop(task_id, None)
            self._task_to_session.pop(task_id, None)
