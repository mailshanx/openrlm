import asyncio
import logging
import json
import time
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Self

from arcgeneral.llm import LLMClient, CompletionResponse

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

This loop repeats — you will get multiple iterations. Do not try to do everything in a single \
python tool call. Break your work into steps: first explore and verify, then build on what works.

When you have the final answer, respond to the user in plain text without calling the python tool.

### Rules
1. The python tool is your ONLY tool. It executes code in a persistent Python REPL. All functions above are called with `await` INSIDE python \
code — never as separate tool calls.
2. If code fails, read the traceback, fix the issue, and retry.
3. Use `asyncio.gather()` to run independent tasks concurrently within a single python tool call.
4. The working directory `{workspace_path}` is shared with the host. Files you create or modify there are visible on the host, and vice versa.
5. If a package is not installed, run `subprocess.run(["uv", "pip", "install", "<package>"])` to install it.
6. To save context space, only your latest user message and its tool interactions are shown in full \
— earlier exchanges are condensed to user message + final response. Your complete history \
including all tool calls, outputs, and errors is available as `_conversation_history` — a list \
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

SUB_AGENT_SUFFIX = (
    "\n\n## Your Role\n"
    "You were created by another agent to accomplish a specific task. "
    "Your final text response (when you stop calling tools) is returned to your creator as the result. "
    "Be specific and structured in your output — it feeds into further work. "
    "For large results, save to disk and return the file path in your response.\n"
    "\n## Additional Instructions\n"
)


@dataclass
class _SubAgent:
    """State for a sub-agent created by the main agent."""
    instructions: str
    depth: int
    sandbox: Sandbox | None = None
    messages: list | None = None
    lock: asyncio.Lock = None  # serializes _run_turn calls per sub-agent

    def __post_init__(self):
        self.lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# RuntimeServices — the narrow interface Session uses to access infrastructure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeServices:
    """Infrastructure capabilities provided by AgentRuntime to each Session.

    Session stores this instead of a back-reference to the full Runtime.
    This is the only coupling between Session and Runtime.
    """
    llm_call: Callable[[list[dict], dict], Awaitable[CompletionResponse]]
    create_sandbox: Callable[[], Awaitable[Sandbox]]
    build_system_message: Callable[[str | None], dict]
    config: AgentConfig
    spool_dir: Path
    spool_path: str  # spool path as visible from inside the sandbox


# ---------------------------------------------------------------------------
# Session — one conversation with its own REPL, message history, sub-agent tree
# ---------------------------------------------------------------------------

class Session:
    """A single conversation with its own REPL, message history, and sub-agent tree.

    Owns the agent loop (_run_turn) and all sub-agent lifecycle. Uses RuntimeServices
    for infrastructure (LLM calls, sandbox creation, message initialization).
    """

    def __init__(self, services: RuntimeServices, sandbox: Sandbox, messages: list,
                 session_id: str, on_event=None):
        self._services = services
        self._sandbox = sandbox
        self._messages = messages
        self._session_id = session_id
        self._on_event = on_event
        self._sub_agents: dict[str, _SubAgent] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    @property
    def session_id(self) -> str:
        """The caller-provided session ID."""
        return self._session_id

    @property
    def request_kwargs(self) -> dict:
        kw: dict = {
            "model": self._services.config.model,
            "tools": [PYTHON_TOOL_SCHEMA],
        }
        if self._services.config.temperature is not None:
            kw["temperature"] = self._services.config.temperature
        return kw

    @property
    def system_message(self) -> dict:
        """The system message for this session, with host function schemas and resolved paths.

        Use this when constructing your own message list for run_turn.
        """
        return self._services.build_system_message(None)

    @property
    def messages(self) -> list[dict]:
        """The session's internal message history.

        This is the live list used by run_single. You may read or modify it
        between turns. The same structural invariants that run_turn enforces
        apply — violations will raise ValueError at the next turn.

        If you want fully independent control, pass your own list to run_turn.
        """
        return self._messages

    # ── Public API ──

    async def run_turn(self, messages: list[dict], user_message: str) -> str:
        """Run the agent loop for a single turn using the provided message list.

        The engine borrows `messages` for the duration of the turn:
        - Appends the user message
        - Appends all assistant and tool messages produced by the loop
        - On cancellation, rolls back everything it appended

        The caller owns `messages` between turns and may mutate it freely,
        subject to structural invariants:
        - First element must be a system message
        - Every assistant message with tool_calls must be followed by
          matching tool result messages
        - The list must not be mutated by another coroutine during the turn

        Args:
            messages: The conversation history. Modified in place.
            user_message: The new user message to process.

        Returns:
            The assistant's final text response.
        """
        async with self._lock:
            messages.append({"role": "user", "content": user_message})
            try:
                self._validate_messages(messages)
            except ValueError:
                messages.pop()
                raise
            try:
                result = await self._run_turn(messages, self._sandbox, agent_label="main")
                return result
            except asyncio.CancelledError:
                for task_id, t in list(self._running_tasks.items()):
                    t.cancel()
                if self._running_tasks:
                    await asyncio.gather(
                        *self._running_tasks.values(), return_exceptions=True,
                    )
                self._running_tasks.clear()
                raise

    async def run_single(self, user_message: str) -> str:
        """Run the agent loop using the session's internal message history.

        Convenience wrapper around run_turn for clients that don't need
        to manage their own message list.
        """
        return await self.run_turn(self._messages, user_message)

    @staticmethod
    def _validate_messages(messages: list[dict]) -> None:
        """Validate structural invariants required by the agent loop.

        Raises ValueError with a specific message on violation.
        """
        if not messages:
            raise ValueError("Message list must not be empty")
        if messages[0].get("role") != "system":
            raise ValueError("First message must have role 'system'")
        if messages[-1].get("role") != "user":
            raise ValueError("Last message must have role 'user' (the turn's input)")

        # Validate tool_calls pairing
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                expected_ids = {tc["id"] for tc in tool_calls}
                found_ids = set()
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    found_ids.add(messages[j].get("tool_call_id"))
                    j += 1
                if found_ids != expected_ids:
                    missing = expected_ids - found_ids
                    extra = found_ids - expected_ids
                    raise ValueError(
                        f"Message {i}: assistant tool_calls not matched by tool messages. "
                        f"Missing: {missing}, unexpected: {extra}"
                    )
                i = j
            else:
                i += 1

    async def close(self):
        """Close this session: cancel tasks, destroy sandboxes.

        Does NOT unregister from Runtime's routing tables — that is
        AgentRuntime.close_session()'s responsibility.
        """
        # Cancel running background tasks
        for task_id, t in list(self._running_tasks.items()):
            t.cancel()
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
        self._sub_agents.clear()

        # Destroy session sandbox
        if self._sandbox is not None:
            try:
                await self._sandbox.close()
            except Exception:
                logger.warning("Failed to destroy session sandbox", exc_info=True)
            finally:
                self._sandbox = None

    # ── Sub-agent management (called by Runtime's host function routing) ──

    def get_agent_depth(self, agent_id: str) -> int:
        """Return the depth of an agent in this session's tree. Root = 0."""
        sub = self._sub_agents.get(agent_id)
        return sub.depth if sub else 0

    def register_sub_agent(self, agent_id: str, instructions: str, depth: int) -> None:
        """Register a new sub-agent in this session's agent tree."""
        logger.info("[session %s] Registered sub-agent %s (depth=%d)", self._session_id, agent_id, depth)
        self._sub_agents[agent_id] = _SubAgent(
            instructions=instructions,
            depth=depth,
        )

    async def start_sub_task(self, agent_id: str, task: str) -> str:
        """Start a background task on a sub-agent. Returns task_id.

        The sub-agent's lock serializes concurrent tasks on the same agent.
        Each task appends a user message and runs a full agent turn, preserving
        the sub-agent's message history and REPL state across tasks.
        """
        sub = self._sub_agents.get(agent_id)
        if sub is None:
            raise RuntimeError(f"Unknown agent_id: {agent_id}")
        task_id = uuid.uuid4().hex[:12]
        logger.info("[session %s] Starting task %s on sub-agent %s: %s",
                     self._session_id, task_id, agent_id, task[:100])

        async def _execute():
            try:
                async with sub.lock:
                    await self._ensure_sub_sandbox(agent_id, sub)
                    sub.messages.append({"role": "user", "content": task})
                    result = await self._run_turn(
                        sub.messages, sub.sandbox,
                        agent_label=agent_id,
                    )
                    logger.info("[session %s] Task %s on sub-agent %s finished",
                                 self._session_id, task_id, agent_id)
                    return result
            except asyncio.CancelledError:
                logger.info("[session %s] Task %s on sub-agent %s cancelled",
                             self._session_id, task_id, agent_id)
                raise

        task_obj = asyncio.create_task(_execute())
        self._running_tasks[task_id] = task_obj
        return task_id

    async def await_sub_task(self, task_id: str) -> str:
        """Block until a sub-agent task completes and return the result."""
        task = self._running_tasks.get(task_id)
        if task is None:
            raise RuntimeError(f"Unknown task_id: {task_id}")
        try:
            return await task
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self._running_tasks.pop(task_id, None)

    # ── Agent loop ──

    async def _run_turn(self, full_history: list, sandbox: Sandbox,
                        agent_label: str = "main") -> str:
        """Run one turn of the agent loop (LLM calls + tool calls until stop).

        Works for both the root agent and sub-agents — the caller passes the
        appropriate messages/sandbox for each.

        On CancelledError, rolls back full_history to the start of the
        interrupted round so the message list stays consistent (every
        assistant tool_calls entry has matching tool results).
        """
        config = self._services.config
        request_kwargs = self.request_kwargs
        on_event = self._on_event
        turn_start = time.monotonic()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for round_num in range(config.max_tool_rounds):
            checkpoint = len(full_history)
            try:
                self._emit(on_event, RoundStart(agent_id=agent_label, round_num=round_num, max_rounds=config.max_tool_rounds))
                compressed = self._compress_messages(full_history)
                self._emit(on_event, ModelRequest(agent_id=agent_label))
                response = await self._services.llm_call(compressed, request_kwargs)

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
                        host_spool_dir=self._services.spool_dir,
                        container_spool_dir=self._services.spool_path,
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

    # ── Internal helpers ──

    async def _ensure_sub_sandbox(self, agent_id: str, sub: _SubAgent) -> None:
        """Lazily create sandbox and init messages on first use."""
        if sub.sandbox is not None:
            return
        logger.info("[session %s] Creating sandbox for sub-agent %s", self._session_id, agent_id)
        sub.sandbox = await self._services.create_sandbox()
        await self._inject_agent_id(sub.sandbox, agent_id)
        sub.messages = [self._services.build_system_message(sub.instructions)]
        logger.info("[session %s] Sub-agent %s ready", self._session_id, agent_id)

    @staticmethod
    async def _inject_agent_id(sandbox: Sandbox, agent_id: str) -> None:
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

    @staticmethod
    async def _sync_history(sandbox: Sandbox, messages: list) -> None:
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


# ---------------------------------------------------------------------------
# AgentRuntime — infrastructure lifecycle + host function routing
# ---------------------------------------------------------------------------

class AgentRuntime:
    """Owns the infrastructure lifecycle: host function server, fork server, LLM client.

    Creates sessions for independent conversations. Routes host function calls
    (create_agent, run_agent, await_result) from any agent at any depth to the
    correct Session via a flat routing table (_agent_to_session).
    """

    def __init__(self, config: AgentConfig, registry: HostFunctionRegistry, llm_client: LLMClient):
        self._config = config
        self._registry = registry
        self._server: HostFunctionServer | None = None
        self._fork_server: ForkServer | None = None
        self._llm_client = llm_client

        # Session tracking
        self._sessions: dict[str, Session] = {}

        # Flat routing tables: every agent_id and task_id at any depth maps
        # back to the root Session that owns its agent tree.
        self._agent_to_session: dict[str, Session] = {}
        self._task_to_session: dict[str, Session] = {}

        # Spool dir for truncated output (temp dir, cleaned up on __aexit__)
        self._spool_dir: Path | None = None

        # Resolved paths (populated in __aenter__, never mutate config)
        self._workspace_path: str | None = None
        self._spool_path: str | None = None

        # RuntimeServices — built lazily in __aenter__ once infrastructure is up
        self._services: RuntimeServices | None = None

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
            self._workspace_path = self._config.workspace_path or "/app/workspace/"
            self._spool_path = self._config.spool_path or "/app/spool"
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
                self._workspace_path = self._config.workspace_path or workspace_host
            else:
                self._workspace_path = self._config.workspace_path or str(Path.cwd())
            self._spool_path = self._config.spool_path or str(self._spool_dir)
        await self._fork_server.__aenter__()

        # Validate API key resolver
        if self._config.get_api_key is None:
            raise ValueError("AgentConfig.get_api_key must be provided")

        # Build the services bundle that Sessions will use
        self._services = RuntimeServices(
            llm_call=lambda msgs, kw: self._llm_call(msgs, kw),
            create_sandbox=self._fork_server.create_sandbox,
            build_system_message=self.build_system_message,
            config=self._config,
            spool_dir=self._spool_dir,
            spool_path=self._spool_path,
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close all sessions (Runtime drives close to clean up routing tables)
        for sid, session in list(self._sessions.items()):
            try:
                await self._close_session_internal(sid, session)
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

        # Close LLM client
        if self._llm_client is not None:
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

        self._services = None

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
        messages = [self.build_system_message()]
        if context_messages:
            messages.extend(context_messages)
        session = Session(
            services=self._services,
            sandbox=sandbox,
            messages=messages,
            session_id=session_id,
            on_event=on_event,
        )
        await Session._inject_agent_id(sandbox, session_id)
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
        """Close and remove a session by ID. No-op if session doesn't exist.

        Cleans up Runtime's routing tables, then closes the Session's own resources.
        """
        session = self._sessions.get(session_id)
        if session is not None:
            await self._close_session_internal(session_id, session)

    async def _close_session_internal(self, session_id: str, session: Session) -> None:
        """Close a session and clean up all routing table entries."""
        # Clean up task routing entries
        for task_id in list(session._running_tasks.keys()):
            self._task_to_session.pop(task_id, None)
        # Clean up agent routing entries
        for agent_id in list(session._sub_agents.keys()):
            self._agent_to_session.pop(agent_id, None)
        self._agent_to_session.pop(session_id, None)
        self._sessions.pop(session_id, None)
        # Now close the session's own resources (tasks, sandboxes)
        await session.close()

    # ── Infrastructure services ──

    def build_system_message(self, extra_instructions: str | None = None) -> dict:
        """Build the system message dict for root or sub-agents.

        For root agents: call with no arguments.
        For sub-agents: engine calls this internally with extra_instructions.
        """
        system_prompt = (
            self._config.system_prompt
            if self._config.system_prompt is not None
            else DEFAULT_SYSTEM_PROMPT
        )
        system_prompt = system_prompt.format(
            functions_json=self._registry.build_schemas_json(),
            workspace_path=self._workspace_path,
            spool_path=self._spool_path,
        )
        if extra_instructions:
            system_prompt += SUB_AGENT_SUFFIX + extra_instructions
        return {"role": "system", "content": system_prompt}

    async def _llm_call(self, messages: list, request_kwargs: dict) -> CompletionResponse:
        """LLM call with retry on transient errors (4 attempts, exponential backoff)."""
        max_attempts = 4
        api_key = await self._config.get_api_key()
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
                api_key = await self._config.get_api_key()

    # ── Host functions (called from kernel via HTTP bridge) ──
    #
    # These are thin routing layers. They look up which Session an agent belongs
    # to via the flat _agent_to_session table, then delegate to Session methods.
    # Every agent_id at every depth maps to the root Session that owns its tree.

    async def _host_create_agent(self, instructions: str, _caller_id: str = "main") -> str:
        """Create a sub-agent with its own Python environment.

        The sub-agent has the same capabilities and functions as you,
        plus any additional instructions you provide.
        Returns an agent_id string to use with run_agent."""
        session = self._agent_to_session.get(_caller_id)
        if session is None:
            raise RuntimeError(f"Unknown caller: {_caller_id}")
        depth = session.get_agent_depth(_caller_id)
        if depth >= self._config.max_sub_agent_depth:
            raise RuntimeError(f"Max sub-agent depth ({self._config.max_sub_agent_depth}) exceeded")
        agent_id = uuid.uuid4().hex[:12]
        session.register_sub_agent(agent_id, instructions, depth + 1)
        self._agent_to_session[agent_id] = session
        return agent_id

    async def _host_run_agent(self, agent_id: str, task: str) -> str:
        """Start a task on a previously created sub-agent.

        Returns a task_id immediately — the sub-agent runs in the background.
        Use await_result(task_id) to collect the result."""
        session = self._agent_to_session.get(agent_id)
        if session is None:
            raise RuntimeError(f"Unknown agent_id: {agent_id}")
        if agent_id not in session._sub_agents:
            raise RuntimeError(f"Unknown agent_id: {agent_id}")
        task_id = await session.start_sub_task(agent_id, task)
        self._task_to_session[task_id] = session
        return task_id

    async def _host_await_result(self, task_id: str) -> str:
        """Block until a background task completes and return the result.

        Pass the task_id returned by run_agent."""
        session = self._task_to_session.get(task_id)
        if session is None:
            raise RuntimeError(f"Unknown task_id: {task_id}")
        if task_id not in session._running_tasks:
            raise RuntimeError(f"Unknown task_id: {task_id}")
        try:
            return await session.await_sub_task(task_id)
        finally:
            self._task_to_session.pop(task_id, None)
