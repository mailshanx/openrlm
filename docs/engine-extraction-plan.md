# Engine Extraction Plan

## Goals

arcgeneral must serve three roles:

1. **A coding agent product** — the CLI and the pi extension.
2. **A library** — importable by Python programs that want an LLM agent with a persistent REPL.
3. **An engine** — an integrated unit that others build their own agent products on top of.

The core value proposition is the engine: an IPython REPL where agents can recursively spawn sub-agents, each with their own REPL, with user-defined host functions callable as plain async Python functions from inside the sandbox. The engine includes:

- The IPython REPL sandbox (fork server, Docker or local mode)
- The agent tool loop (`_run_turn` — LLM call, tool dispatch, repeat until stop)
- Recursive sub-agents (create/run/await, routing tables, depth enforcement)
- The host function bridge (registry, HTTP server, kernel stub generation)
- Infrastructure wiring (server lifecycle, sandbox creation, preamble injection, cleanup)
- Sub-agent prompt wiring (the "Your Role" suffix and `extra_instructions` injection)

These are one integrated unit. Users do not rewrite the agent loop, replace sub-agent execution, or manage fork servers. They get the engine as a whole.

What users plug in:

- **An LLM client** — any implementation of the `LLMClient` protocol. The engine never auto-selects a provider.
- **A system prompt template** — a string with `{functions_json}`, `{workspace_path}`, `{spool_path}` placeholders. The engine has a good default (`DEFAULT_SYSTEM_PROMPT`). The engine handles sub-agent instructions itself — the user's template applies identically to root and sub-agents.
- **Custom host functions** — registered into `HostFunctionRegistry`, appear as async callables in every agent's REPL.
- **Config knobs** — max rounds, max depth, timeouts, model name, temperature, etc.
- **Conversation history** — the client may own the message list and pass it to each turn, or let the engine manage it internally. The engine borrows the list during a turn and returns it enriched.

What users do not plug in:

- The agent loop. `_run_turn` is core. Sub-agents run the same loop as the root agent, with the same LLM client, same tools, same infrastructure.
- Sandbox or fork server management. The engine handles this.
- Sub-agent routing. The engine handles this.
- Sub-agent prompt wiring. The engine appends the "Your Role" suffix and `extra_instructions` to sub-agent prompts. The user's template does not need to account for this.
- Sub-agent message histories. Sub-agent conversations are engine-internal. The client controls only the root agent's messages.

## What constitutes core

Core is everything the engine owns. The test: "could a user reasonably want to swap this out when building a different product?" If no, it's core.

| Component | Core? | Rationale |
|---|---|---|
| `_run_turn` (agent loop) | Yes | The loop shape is the product. |
| Sub-agent routing tables | Yes | Inseparable from the loop — sub-agents run the same loop. |
| `create_agent`/`run_agent`/`await_result` host functions | Yes | Part of the recursive agent model. Not user-replaceable. |
| Sub-agent prompt suffix (`SUB_AGENT_SUFFIX` + `extra_instructions`) | Yes | The engine controls how sub-agents receive their role and instructions. |
| `HostFunctionRegistry` + `HostFunctionServer` | Yes | The bridge between sandbox and host. |
| `ForkServer` / `LocalForkServer` / `Sandbox` | Yes | REPL lifecycle. |
| `execute_tool` + `PYTHON_TOOL_SCHEMA` | Yes | The only tool is Python. Host functions are called from within Python. |
| Message compression (`_compress_messages`) | Yes | Cost control for the loop. |
| Message validation (`_validate_messages`) | Yes | Protects the engine from malformed input at the `run_turn` boundary. |
| History sync (`_sync_history`) | Yes | Agents can inspect their own history. |
| Events (`AgentEvent` types) | Yes | Observation mechanism for the loop. |
| `LLMClient` protocol + response types | Yes | The interface. Not any specific implementation. |
| `DEFAULT_SYSTEM_PROMPT` | Bundled default | Ships with the engine. Replaceable with another template string. |
| Root agent message history | Client-owned | The client may manage it or let the engine manage it via `run_single`. |
| `AnthropicClient` / `OpenRouterClient` | No | Specific provider implementations. Pluggable. |
| `default_api_key_resolver` / `_read_auth_file` / `PROVIDER_ENV_VARS` | No | Auth utilities for constructing specific clients. |
| LLM client auto-selection (the if/else in `__aenter__`) | No | Product-level decision. Belongs in CLI/extension. |
| `AgentConfig.provider` | No | An LLM client construction concern, not an engine concern. |
| CLI (`cli.py`) | No | Harness. |

## Design principle: the engine owns execution, the client owns conversation

The engine and the client meet at `run_turn`. The client hands in a structurally valid message list and a user message. The engine borrows the list for the duration of the turn — appending the user message, all assistant and tool messages from the loop, and the final response. On cancellation, the engine rolls back everything it appended. When the turn ends, the client has the list back and may mutate it freely before the next turn.

This separates two concerns:

- **Execution context** (engine-owned): the sandbox process, REPL state, sub-agent tree, host function stubs. Created at session creation, destroyed at session close. Not transferable.
- **Conversation state** (client-owned or engine-managed): the message list. The client can construct it, inject synthetic exchanges, truncate old turns, fork it across sessions, or let the engine accumulate it internally via `run_single`.

`run_single` becomes a convenience wrapper that uses an engine-internal message list. Clients who need control use `run_turn` directly with their own list.

Sub-agent message histories remain engine-internal. The client controls only the root agent's conversation. Sub-agents are spawned by the root agent's code and managed entirely by the engine.

## Changes

### 1. Make `llm_client` required on `AgentRuntime`

**Current:** `AgentRuntime.__init__` accepts `llm_client: LLMClient | None = None`. In `__aenter__`, if None, it auto-selects based on `config.provider`:

```python
if self._config.provider == "anthropic":
    self._llm_client = AnthropicClient()
else:
    self._llm_client = OpenRouterClient()
```

**Change:** Make `llm_client` a required argument. Remove the auto-selection. Remove the `OpenRouterClient` and `AnthropicClient` imports from `agent.py`.

```python
def __init__(self, config: AgentConfig, registry: HostFunctionRegistry, llm_client: LLMClient):
```

If `llm_client` is None, raise `TypeError` at construction, not silently at `__aenter__`.

**Migrate callers:**
- `cli.py` `main()`: construct the LLM client based on `args.provider` before creating the runtime.
- Tests: pass explicit mock or real clients.

### 2. Remove `provider` from `AgentConfig`

**Current:** `AgentConfig` has `provider: str = "openrouter"` and `get_api_key: Callable[[str], Awaitable[str]] | None`. The engine calls `await config.get_api_key(config.provider)` in `_llm_call`.

**Change:** Replace with `get_api_key: Callable[[], Awaitable[str]] | None` — a zero-argument callable. The provider is baked in by the caller when constructing the closure. Remove the `provider` field entirely.

```python
@dataclass
class AgentConfig:
    model: str = "openai/gpt-5.2"
    get_api_key: Callable[[], Awaitable[str]] | None = None
    system_prompt: str | None = None
    sandbox_image: str | None = None
    code_timeout: float = 3600.0
    max_tool_rounds: int = 50
    max_sub_agent_depth: int = 10
    output_limit_lines: int = 2000
    output_limit_bytes: int = 50_000
    temperature: float | None = None
    sandbox_binds: dict[str, str] = field(default_factory=dict)
    workspace_path: str | None = None
    spool_path: str | None = None
```

**Migrate `_llm_call`:**

```python
# Before:
api_key = await self._config.get_api_key(self._config.provider)

# After:
api_key = await self._config.get_api_key()
```

**Migrate `cli.py`:**

```python
# Before:
config = AgentConfig(provider=args.provider, model=args.model, ...)

# After:
resolver = default_api_key_resolver()
config = AgentConfig(
    model=args.model,
    get_api_key=lambda: resolver(args.provider),
    ...
)
```

### 3. Keep system prompt as `str | None` — no callable

**Current:** `AgentConfig.system_prompt` is `str | None`. If None, `DEFAULT_SYSTEM_PROMPT` is used. The string is formatted with `{functions_json}`, `{workspace_path}`, `{spool_path}`. For sub-agents, the engine appends a "Your Role" preamble and the `extra_instructions` string from `create_agent()`.

**Change:** No change to the type or mechanism. Extract the inline sub-agent suffix into a named constant for readability:

```python
SUB_AGENT_SUFFIX = (
    "\n\n## Your Role\n"
    "You were created by another agent to accomplish a specific task. "
    "Your final text response (when you stop calling tools) is returned to your creator as the result. "
    "Be specific and structured in your output — it feeds into further work. "
    "For large results, save to disk and return the file path in your response.\n"
    "\n## Additional Instructions\n"
)
```

A user who wants a different system prompt provides a template string with the same placeholders. The engine formats it and handles sub-agent instructions. Root and sub-agents get the same base prompt; sub-agents additionally get the suffix and their `extra_instructions`.

### 4. Move auth utilities out of the engine import path

**Current:** `agent.py` imports `default_api_key_resolver` from `llm.py` and calls it in `__aenter__` as a fallback when `config.get_api_key` is None.

**Change:** Remove the `default_api_key_resolver` call from `AgentRuntime.__aenter__`. If `config.get_api_key` is None at `__aenter__` time, raise an error. The caller is responsible for providing it.

`default_api_key_resolver`, `_read_auth_file`, and `PROVIDER_ENV_VARS` stay in `llm.py` as public utilities. They are used by `cli.py` and the pi extension when constructing configs. The engine does not call them.

Remove the `OpenRouterClient` and `AnthropicClient` imports from `agent.py`. `agent.py` only imports the `LLMClient` protocol and `CompletionResponse` types from `llm.py`.

### 5. Update `cli.py` to construct LLM client and auth

**Current:** `cli.py` creates `AgentConfig(provider=args.provider, model=args.model)` and lets `AgentRuntime` handle the rest.

**Change:** `cli.py` constructs the LLM client and the API key resolver based on `--provider`:

```python
from arcgeneral.llm import (
    LLMClient, OpenRouterClient, AnthropicClient,
    default_api_key_resolver,
)

def _build_llm_client(provider: str) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient()
    else:
        return OpenRouterClient()

def main():
    args = parse_args()
    ...
    resolver = default_api_key_resolver()
    config = AgentConfig(
        model=args.model,
        get_api_key=lambda: resolver(args.provider),
        sandbox_image=args.image,
        code_timeout=args.timeout,
        max_tool_rounds=args.max_rounds,
        ...
    )
    llm_client = _build_llm_client(args.provider)
    runtime = AgentRuntime(config, registry, llm_client=llm_client)
    ...
```

### 6. Update `__init__.py` exports

**Current:** Exports everything flat: event types, LLM client implementations, response types, AgentRuntime, Session, config, registry, sandbox types.

**Change:** No additions or removals. All current exports remain. The `LLMClient` protocol, `CompletionResponse` types, `OpenRouterClient`, `AnthropicClient`, `default_api_key_resolver`, and `PROVIDER_ENV_VARS` remain exported as public utilities for constructing LLM clients. They are no longer imported or used by the engine (`agent.py`) itself.

### 7. Stop mutating `AgentConfig` in `__aenter__`

**Current:** `AgentRuntime.__aenter__` mutates `config.workspace_path`, `config.spool_path`, and `config.get_api_key`.

**Change:** Resolve these into instance variables on `AgentRuntime`. Do not mutate the config object. The config the user passed in remains unchanged.

```python
async def __aenter__(self):
    ...
    # Resolve paths (don't mutate config)
    if self._config.sandbox_image:
        self._workspace_path = self._config.workspace_path or "/app/workspace/"
        self._spool_path = self._config.spool_path or "/app/spool"
    else:
        ...
        self._workspace_path = self._config.workspace_path or str(Path.cwd())
        self._spool_path = self._config.spool_path or str(self._spool_dir)
```

### 8. Extract `request_kwargs` from session state to a computed property

**Current:** `_init_messages` returns a `(messages, request_kwargs)` tuple. `request_kwargs` is `{"model": ..., "tools": [PYTHON_TOOL_SCHEMA], "temperature": ...}`, computed once at session creation and stored as `self._request_kwargs` on `Session` and `sub.request_kwargs` on `_SubAgent`. It's threaded through `_run_turn` as a parameter. `RuntimeServices.init_messages` exists solely to let sub-agents call `_init_messages`.

**Change:** Make `request_kwargs` a computed property on `Session`, derived from config:

```python
@property
def request_kwargs(self) -> dict:
    kw: dict = {
        "model": self._services.config.model,
        "tools": [PYTHON_TOOL_SCHEMA],
    }
    if self._services.config.temperature is not None:
        kw["temperature"] = self._services.config.temperature
    return kw
```

Remove `request_kwargs` from:
- `Session.__init__` parameters and instance state
- `_SubAgent` fields
- `_run_turn` parameters — it reads `self.request_kwargs` directly
- `RuntimeServices.init_messages` — replaced by `build_system_message` (change 9)
- `AgentRuntime.create_session` — no longer computes or passes it

`_run_turn` for sub-agents also uses `self.request_kwargs` since sub-agents share the same model and tools as the root agent. This is already the case today — `request_kwargs` is identical for root and sub-agents — but the current code redundantly stores separate copies.

### 9. Replace `_init_messages` with `build_system_message`

**Current:** `AgentRuntime._init_messages` builds both the system message and `request_kwargs`, returning `(messages, request_kwargs)`. It's exposed through `RuntimeServices.init_messages` so sub-agents can call it in `_ensure_sub_sandbox`. After change 8, `request_kwargs` is gone from this path.

**Change:** Replace with a public method that builds only the system message:

```python
# On AgentRuntime:
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
```

Update `RuntimeServices`:

```python
@dataclass(frozen=True)
class RuntimeServices:
    llm_call: Callable[[list[dict], dict], Awaitable[CompletionResponse]]
    create_sandbox: Callable[[], Awaitable[Sandbox]]
    build_system_message: Callable[[str | None], dict]
    config: AgentConfig
    spool_dir: Path
```

Update `_ensure_sub_sandbox`:

```python
# Before:
sub.messages, sub.request_kwargs = self._services.init_messages(
    self._services.config, extra_instructions=sub.instructions,
)

# After:
sub.messages = [self._services.build_system_message(sub.instructions)]
```

Update `AgentRuntime.create_session`:

```python
# Before:
messages, request_kwargs = self._init_messages(self._config)

# After:
messages = [self.build_system_message()]
```

This makes `build_system_message()` a public method. Clients constructing their own message list call it to get the correctly formatted system message with host function schemas and resolved paths.

### 10. Add `run_turn` as the primary public API on `Session`

**Current:** `Session.run_single` is the only public entry point. It appends the user message to `self._messages`, calls `_run_turn`, and returns the result. The client has no way to provide their own message list.

**Change:** Add `run_turn` as a new method that accepts a client-provided message list:

```python
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
```

Add `_validate_messages` as a static method on `Session`:

```python
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
```

### 11. Rewrite `run_single` as a wrapper around `run_turn`

**Current:** `run_single` contains the full turn execution logic: append user message, call `_run_turn`, handle cancellation.

**Change:** `run_single` delegates to `run_turn` using the session's internal message list:

```python
async def run_single(self, user_message: str) -> str:
    """Run the agent loop using the session's internal message history.

    Convenience wrapper around run_turn for clients that don't need
    to manage their own message list.
    """
    return await self.run_turn(self._messages, user_message)
```

Existing callers of `run_single` — the CLI, the pi extension, tests — continue to work unchanged.

### 12. Expose `system_message` and `messages` properties on `Session`

**Current:** `self._messages` is private. There is no way to read or modify the session's message history from outside.

**Change:** Add read-only properties:

```python
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
```

`system_message` rebuilds the system message from scratch (it's string formatting — cheap). This ensures the client always gets a correctly formatted message even if they construct their list long after session creation.

`messages` exposes the live list reference. The client can inspect it, append synthetic exchanges, or truncate old turns. Structural invariants are enforced by `_validate_messages` at the top of each `run_turn` call, so corruption is caught at the boundary rather than producing opaque LLM API errors.

## Execution order

These changes should be applied in this order to keep the codebase working at each step:

1. **Stop mutating config** (change 7) — mechanical, no API change.
2. **Make `llm_client` required** (change 1) — update `AgentRuntime.__init__` signature.
3. **Update `cli.py`** (change 5) — construct client in CLI, pass to runtime. Tests pass again.
4. **Remove `provider` from config** (change 2) — change `get_api_key` signature, update `_llm_call` and CLI.
5. **Move auth imports out of `agent.py`** (change 4) — remove `OpenRouterClient`/`AnthropicClient`/`default_api_key_resolver` imports from `agent.py`.
6. **Extract `SUB_AGENT_SUFFIX` constant** (change 3) — extract the inline string, no behavior change.
7. **Update exports** (change 6) — verify no additions or removals needed.
8. **Extract `request_kwargs` to property** (change 8) — remove from `Session.__init__`, `_SubAgent`, `_run_turn` params. Add computed property.
9. **Replace `_init_messages` with `build_system_message`** (change 9) — update `RuntimeServices`, `create_session`, `_ensure_sub_sandbox`.
10. **Add `run_turn` and `_validate_messages`** (change 10) — new public primitive.
11. **Rewrite `run_single`** (change 11) — becomes a one-line wrapper.
12. **Expose properties** (change 12) — `system_message`, `messages`, `request_kwargs`.

## Client example: building a custom agent harness

This example shows a research assistant product built on the engine. It demonstrates every integration point: custom LLM client, custom host functions, client-owned message history with injected context between turns, event observation, and multi-session management.

```python
import asyncio
from arcgeneral import (
    AgentRuntime, AgentConfig, HostFunctionRegistry, Session,
    LLMClient, CompletionResponse, CompletionChoice, CompletionMessage,
    ToolCall, ToolCallFunction, TokenUsage,
    AgentEvent, TurnEnd, ModelResponse,
)


# ── 1. Implement LLMClient ──────────────────────────────────────────────

class GeminiClient:
    """Custom LLM client wrapping the Gemini API."""

    def __init__(self):
        import google.generativeai as genai
        self._genai = genai

    async def complete(self, messages: list[dict], *, api_key: str, **kwargs) -> CompletionResponse:
        model_name = kwargs.get("model", "gemini-2.5-pro")
        # ... translate messages to Gemini format, call API, translate response
        # back into CompletionResponse with CompletionChoice, CompletionMessage,
        # ToolCall/ToolCallFunction (if tool use), and TokenUsage.
        ...

    async def close(self) -> None:
        pass


# ── 2. Register custom host functions ───────────────────────────────────

registry = HostFunctionRegistry()

async def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv for papers matching the query. Returns titles, authors, and abstracts."""
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://export.arxiv.org/api/query", params={
            "search_query": f"all:{query}", "max_results": max_results,
        })
    return resp.text

async def save_finding(title: str, summary: str, source: str) -> str:
    """Save a research finding to the project database."""
    # ... insert into database
    return f"Saved: {title}"

registry.register("arxiv_search", arxiv_search, timeout=30.0)
registry.register("save_finding", save_finding)


# ── 3. Configure the engine ─────────────────────────────────────────────

config = AgentConfig(
    model="gemini-2.5-pro",
    get_api_key=lambda: vault.get_secret("gemini-key"),
    system_prompt=(
        "You are a research assistant with access to arXiv and a findings database.\n\n"
        "Available functions:\n{functions_json}\n\n"
        "Working directory: {workspace_path}\n"
        "Spool directory: {spool_path}\n\n"
        "Use the Python REPL to analyze papers, extract data, and save findings."
    ),
    max_tool_rounds=30,
    max_sub_agent_depth=3,
    sandbox_binds={"/data/research": "workspace"},
)


# ── 4. Run with client-owned message history ────────────────────────────

async def research_session():
    client = GeminiClient()
    runtime = AgentRuntime(config, registry, llm_client=client)

    async with runtime:
        session = await runtime.create_session("research-1", on_event=log_event)

        # Client constructs and owns the message list
        messages = [session.system_message]

        # Turn 1: initial research request
        result = await session.run_turn(
            messages, "Find recent papers on sparse attention mechanisms"
        )
        # messages now contains: [system, user, assistant+tools, ..., assistant_final]
        print(f"Turn 1: {result[:200]}")

        # Between turns: inject context from an external system.
        # This is why client-owned messages matter — the engine can't know
        # about events happening outside the agent loop.
        messages.append({
            "role": "user",
            "content": "UPDATE: Our team just published a related paper. "
                       "See /data/research/our_paper.pdf for context.",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I'll factor in your team's paper when "
                       "analyzing the results.",
        })

        # Turn 2: follow-up that builds on both the REPL state from turn 1
        # AND the injected context
        result2 = await session.run_turn(
            messages, "Compare our approach to the top 3 papers you found"
        )
        print(f"Turn 2: {result2[:200]}")

        # The messages list has the full history — persist it, fork it, etc.
        save_conversation_to_db("research-1", messages)

        await runtime.close_session("research-1")


# ── 5. Observe events for billing and monitoring ────────────────────────

total_cost = 0.0

def log_event(event: AgentEvent):
    global total_cost
    match event:
        case ModelResponse(agent_id=aid, prompt_tokens=pt, completion_tokens=ct):
            cost = (pt or 0) * 0.001 + (ct or 0) * 0.003
            total_cost += cost
            print(f"  [{aid}] LLM call: {pt}+{ct} tokens (${cost:.4f})")
        case TurnEnd(agent_id=aid, rounds=r, elapsed_seconds=t):
            print(f"  [{aid}] Turn done: {r} rounds, {t:.1f}s")


# ── 6. Alternative: use run_single for simple cases ─────────────────────

async def quick_question():
    """For simple use cases, run_single manages messages internally."""
    client = GeminiClient()
    runtime = AgentRuntime(config, registry, llm_client=client)

    async with runtime:
        session = await runtime.create_session("quick")
        # No message list management needed
        result = await session.run_single("How many arXiv papers mention 'mamba' in 2024?")
        print(result)
        await runtime.close_session("quick")


# ── 7. Multi-session: parallel independent research tracks ──────────────

async def parallel_research():
    client = GeminiClient()
    runtime = AgentRuntime(config, registry, llm_client=client)

    async with runtime:
        # Two independent sessions with separate sandboxes and message histories
        session_a = await runtime.create_session("track-a")
        session_b = await runtime.create_session("track-b")

        msgs_a = [session_a.system_message]
        msgs_b = [session_b.system_message]

        # These run concurrently — different sandboxes, no shared state
        result_a, result_b = await asyncio.gather(
            session_a.run_turn(msgs_a, "Survey transformer efficiency papers from 2024"),
            session_b.run_turn(msgs_b, "Survey state-space model papers from 2024"),
        )

        # Merge findings in a third session
        session_c = await runtime.create_session("synthesis")
        msgs_c = [session_c.system_message]
        combined = await session_c.run_turn(
            msgs_c,
            f"Synthesize these two survey results into a comparison:\n\n"
            f"TRANSFORMERS:\n{result_a}\n\nSTATE-SPACE:\n{result_b}",
        )
        print(combined)

        for sid in ("track-a", "track-b", "synthesis"):
            await runtime.close_session(sid)
```

### What the client controls

- **Which LLM to call**: `GeminiClient` implements the `LLMClient` protocol. Could be any provider, a routing layer, or a mock.
- **Authentication**: `get_api_key` is a zero-argument callable. The client bakes in their secret management.
- **Host functions**: `arxiv_search` and `save_finding` appear as `await arxiv_search(...)` inside every agent's REPL. The engine generates stubs and schemas automatically from type hints.
- **System prompt**: custom template with `{functions_json}`, `{workspace_path}`, `{spool_path}` placeholders. The engine formats it and handles sub-agent instructions.
- **Message history**: the client constructs the list from `session.system_message`, passes it to `run_turn`, and freely modifies it between turns. The engine validates structural invariants at the start of each turn.
- **Session lifecycle**: create, use, close. Multiple sessions run concurrently with isolated sandboxes.
- **Event observation**: `on_event` callback receives typed events for logging, billing, UI updates.

### What the engine controls

- **The agent loop**: LLM call, tool dispatch, repeat. Same loop for root and sub-agents.
- **Sandbox lifecycle**: fork servers, process management, preamble injection, cleanup.
- **Sub-agent tree**: `create_agent`/`run_agent`/`await_result` routing, depth enforcement, per-agent locks.
- **Sub-agent messages**: engine-internal, never exposed to the client.
- **Host function bridge**: HTTP server, kernel stubs, JSON schema generation.
- **Message compression**: prior turns condensed for the LLM; the client's list stays full-fidelity.
- **History sync**: `_conversation_history` in the kernel reflects the current state after each round.
- **Cancellation rollback**: on `CancelledError`, the engine rolls back everything it appended during the interrupted turn.
