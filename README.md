# arcgeneral

A recursive language model (RLM) agent harness with persistent IPython REPL environments. Usable as a CLI, embedded in an existing harness, or as a library.

Each agent gets a stateful IPython environment where it can persist variables, define functions, and run computations across multiple turns. Agents can programmatically spawn sub-agents, each with their own isolated REPL, to arbitrary depth.

## Why

**Why RLM?** The [Recursive Language Model](https://arxiv.org/abs/2512.24601) paper from MIT shows that recursive decomposition significantly improves performance on long-context and complex reasoning tasks. An agent that can spawn sub-agents to handle sub-problems — each with their own scratch space — outperforms flat agent loops.

**Why this implementation?** The original RLM implementation treats the user prompt as a variable the LLM greps and chunks. In practice, you want agents to operate on files and data in an application-specific context with custom tools. This implementation provides:

- **Custom host functions.** Define tools (search, APIs, domain-specific operations) that execute on the host but appear as plain async Python functions inside the agent's REPL. Serialization is invisible to the LLM.
- **Persistent REPL state.** Agents persist data to an IPython environment they can access across turns — variables, imports, and function definitions all survive between tool calls. Some ARC-AGI implementations demonstrated superior performance with this pattern, but lacked recursive sub-agents.
- **Cheap sub-agent spawning.** Sub-agents are forked processes. The fork server pre-imports expensive packages (numpy, pandas, etc.), then calls `gc.freeze()` before forking. Children inherit all imported modules via copy-on-write pages, and `gc.freeze()` prevents the garbage collector from scanning those objects — which would dirty the pages and force real memory copies. The OS only allocates memory for new data each sub-agent creates. A single machine can support hundreds to thousands of concurrent sub-agents.

## Architecture
arcgeneral has two layers. The **core** gives an LLM a persistent IPython REPL with host function injection. The **harness** adds a managed agent loop, message history, compression, and recursive sub-agent orchestration on top — including the `create_agent`/`run_agent`/`await_result` functions that enable sub-agent spawning.

### Core: Fork Server + Sandbox + Host Functions

The core handles execution: you make an LLM call, extract the code from the response, and run it in a sandbox. State persists across `execute()` calls. Host functions you register appear as async callables inside the REPL. You control the LLM provider, the prompt, and what happens with the output.

```python
from arcgeneral import LocalForkServer, HostFunctionRegistry

registry = HostFunctionRegistry()
# optionally register host functions on the registry

async with LocalForkServer(registry) as server:
    sandbox = await server.create_sandbox()

    # You make your own LLM call, get code back, execute it
    code = await my_llm_call("compute the mean of [1, 2, 3] using numpy")
    output = await sandbox.execute(code, timeout=30)
    print(output)  # "2.0"

    # Sandbox persists state — variables survive across execute() calls
    output = await sandbox.execute("print(x + 1)", timeout=30)  # x was set by prior code

    await sandbox.close()
```

The fork server pre-imports expensive packages once, calls `gc.freeze()`, then forks a child process for each sandbox. Children share pre-imported module memory via OS-level copy-on-write. Host functions registered on the caller side are injected as async stubs into each sandbox's REPL — the code inside calls `await my_function(...)` and it transparently round-trips to the host via HTTP.

Both local mode (subprocess, the default) and Docker mode (container, for isolation) use the same TCP-based protocol. The LLM provider is entirely your choice at this layer.

To build your own harness on the core, you would:
1. Create a `HostFunctionRegistry` and register any domain-specific functions
2. Start a `LocalForkServer` (or `ForkServer` for Docker) with the registry
3. Call `server.create_sandbox()` to get a `Sandbox` for each agent
4. Make your own LLM calls, extract code, run it via `sandbox.execute(code, timeout)`
5. To support sub-agents, register your own `create_agent`/`run_agent`/`await_result` host functions that create additional sandboxes and route tasks between them — this is what `AgentRuntime` does internally

### Harness: Session + AgentRuntime

The harness wraps the core with LLM-driven automation:

```
+----------------------------------------------------------+
|  AgentRuntime                                            |
|                                                          |
|  +------------+  +------------------+                    |
|  | LLM Client |  | Host Function    |                    |
|  | (OpenRouter |  | Server           |                    |
|  |  or custom) |  | (HTTP bridge)    |                    |
|  +------------+  +--------+---------+                    |
|                           |                              |
|  +------------------------|-----------------------+      |
|  | Session                |                       |      |
|  |                        |                       |      |
|  |  +--------------+      |                       |      |
|  |  | Agent Loop   |      |                       |      |
|  |  | (LLM <> REPL)|      |                       |      |
|  |  +------+-------+      |                       |      |
|  |         |              |                       |      |
|  |         v              |                       |      |
|  |  +---------------+     |                       |      |
|  |  | Sandbox       |<----+                       |      |
|  |  | (IPython REPL)|  host function stubs        |      |
|  |  +------+--------+                             |      |
|  |         | create_agent / run_agent              |      |
|  |         v                                      |      |
|  |  +-------------+  +-------------+              |      |
|  |  | Sub-agent   |  | Sub-agent   |  ...         |      |
|  |  | (own REPL)  |  | (own REPL)  |              |      |
|  |  +-------------+  +-------------+              |      |
|  +------------------------------------------------+      |
|                                                          |
|  +----------------------------------------------------+  |
|  | Fork Server                                        |  |
|  | (local subprocess or Docker container)             |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
```

**AgentRuntime** manages shared infrastructure: the LLM client, the host function server, the fork server, and output spooling. It creates Sessions and routes sub-agent operations from any depth in the agent tree to the correct Session.

**Session** owns a single conversation: the agent loop (LLM calls + tool execution), message history, the root sandbox, and the full sub-agent tree. You call `session.run_single("message")` to run a turn. For multi-turn use, call it repeatedly — messages and REPL state accumulate across turns. Each Session is independent; multiple Sessions can run concurrently on the same Runtime.

If you need full control over message construction, LLM calls, or tool dispatch, use the core layer directly and build your own harness (see the core section above).

**Two execution modes:**

- **Local (default):** Fork server runs as a subprocess on the host machine. No Docker required. The workspace directory defaults to your current working directory — files agents create there appear on your filesystem.
- **Docker:** Fork server runs inside a container for isolation. Host directories are exposed via bind mounts. Use `--image` to enable.

## Installation

```bash
pip install arcgeneral
# or
uv pip install arcgeneral
```

To use the bundled internet search/extract tools:

```bash
pip install arcgeneral[contrib]
```

To use Docker mode, build the sandbox image:

```python
from arcgeneral.ipybox.build import build
from pathlib import Path
build("arcgeneral:sandbox", Path("sandbox-deps.txt"))
```

## Quickstart

### CLI

```bash
# Single message (local mode, default)
arcgeneral "compute the first 20 prime numbers"

# Interactive session
arcgeneral

# With a specific model (routed through OpenRouter)
arcgeneral --model anthropic/claude-sonnet-4-5 "explain main.py"

# With custom tools
arcgeneral --functions ./my-tools "use my_search to find X"

# With bundled contrib tools (requires PARALLEL_API_KEY)
arcgeneral --functions ./contrib "search for recent advances in fusion energy"

# Docker mode
arcgeneral --image arcgeneral:sandbox "analyze data"

# JSON output for programmatic use
arcgeneral --json "compute pi to 50 digits" | jq .result

# With conversation context from a prior session
arcgeneral --context history.json "continue the analysis"
```

### Library

#### Single-shot

```python
import asyncio
from arcgeneral import AgentRuntime, AgentConfig, HostFunctionRegistry

async def main():
    config = AgentConfig(
        model="openai/gpt-5.2",
        provider="openrouter",
    )
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry)

    async with runtime:
        session = await runtime.create_session("my-session")
        result = await session.run_single("What is 2 + 2?")
        print(result)
        await runtime.close_session("my-session")

asyncio.run(main())
```

#### Multi-turn
Call `run_single()` repeatedly on the same session. The Session manages the full conversation history internally — you don't construct or pass messages. Each call appends a user message, runs the agent loop (which may involve multiple LLM/tool-call rounds), and returns the final assistant response as a string. REPL state (variables, imports, computed results) also persists across turns.
```python
async with runtime:
    session = await runtime.create_session("analysis")
    # Turn 1: agent loads data, stores DataFrame in a REPL variable
    response = await session.run_single("Load data.csv and show me the column names")
    print(response)  # "The file has columns: date, product, price, volume ..."

    # Turn 2: agent reuses the loaded DataFrame — no re-reading needed
    response = await session.run_single("What's the correlation between price and volume?")
    print(response)  # "The Pearson correlation is 0.73 ..."

    # Turn 3: agent builds on all prior computed state
    response = await session.run_single("Plot the top 5 outliers and save to outliers.png")
    print(response)  # "Saved outliers.png with 5 data points highlighted ..."
    await runtime.close_session("analysis")
```

The caller's only job is to provide the user message and consume the response string. Everything else — message accumulation, compression of older turns, tool call execution, history synchronization to the REPL — happens inside the Session.

### Custom Host Functions
Define tools that execute on the host but appear as native async functions inside the agent's REPL.

#### Library usage

When using arcgeneral as a library, you create a `HostFunctionRegistry`, register functions on it, and pass it to `AgentRuntime`:

```python
import json
from arcgeneral import AgentRuntime, AgentConfig, HostFunctionRegistry

async def my_database_query(sql: str, limit: int = 100) -> str:
    """Execute a SQL query against the application database.

    Returns results as a JSON string."""
    results = await db.execute(sql, limit=limit)
    return json.dumps(results)

registry = HostFunctionRegistry()
registry.register("my_database_query", my_database_query)

# The registry is passed to the runtime, which injects the functions into every agent's REPL
runtime = AgentRuntime(AgentConfig(), registry)
```

Inside the agent's REPL, the function becomes callable as:

```python
result = await my_database_query(sql="SELECT * FROM users", limit=10)
```

The function's type hints and docstring are picked up automatically — Pydantic builds a JSON schema from the signature for the system prompt, and the docstring becomes the description the LLM sees. No separate schema definitions needed.

#### CLI usage (`--functions`)

When using the CLI, you don't create a registry yourself — the CLI creates one and needs a way to discover your functions. You provide a Python file (or directory of files) that exports a `register(registry)` function. The CLI calls it, passing its own `HostFunctionRegistry` instance:

```python
# my_tools.py
import json

async def my_database_query(sql: str, limit: int = 100) -> str:
    """Execute a SQL query against the application database.

    Returns results as a JSON string."""
    results = await db.execute(sql, limit=limit)
    return json.dumps(results)

def register(registry):
    """Called by the CLI with its HostFunctionRegistry. Register your functions here."""
    registry.register("my_database_query", my_database_query)
```

Then:

```bash
arcgeneral --functions my_tools.py "show me the top 10 users"
```

For a directory of tool files, each `.py` file with a `register()` function is loaded automatically (files starting with `_` are skipped):

```bash
arcgeneral --functions ./my-tools/ "analyze the data"
```

You can also use a dotted module name for installed packages:

```bash
arcgeneral --functions my_package.tools "do something"
```

### Event Streaming

Monitor agent activity with event callbacks. Events from sub-agents at any depth flow through the same callback, distinguished by `agent_id`:

```python
from arcgeneral import AgentRuntime, AgentConfig, HostFunctionRegistry
from arcgeneral.events import (
    RoundStart, ModelResponse, ToolExecStart, ToolExecEnd, TurnEnd,
)

def on_event(event):
    match event:
        case RoundStart(agent_id=aid, round_num=n):
            print(f"[{aid}] Round {n}")
        case ToolExecEnd(agent_id=aid, elapsed_seconds=t):
            print(f"[{aid}] Tool execution: {t:.1f}s")
        case TurnEnd(agent_id=aid, rounds=r, prompt_tokens=pt, completion_tokens=ct):
            print(f"[{aid}] Done in {r} rounds, {pt}+{ct} tokens")

async with runtime:
    session = await runtime.create_session("s1", on_event=on_event)
    await session.run_single("analyze this dataset")
```

## Sub-agents

Agents can spawn sub-agents programmatically from within the REPL:

```python
# Create a sub-agent with specific instructions
agent_id = await create_agent(instructions="You are a citation specialist")

# Start a task (non-blocking — runs in the background)
task_id = await run_agent(agent_id=agent_id, task="Research citation percentiles for federal courts")

# Do other work while sub-agent runs...

# Collect the result
result = await await_result(task_id)
```

Sub-agents can themselves spawn sub-agents, enabling recursive decomposition. Each sub-agent has:
- Its own isolated IPython namespace
- Its own conversation history with the LLM
- Access to the same host functions and shared workspace directory
- A per-agent lock that serializes concurrent tasks on the same sub-agent

**Persistent sub-agents.** A sub-agent created with `create_agent` persists across multiple `run_agent` calls. Each task appends a new user message and runs a full agent turn, so the sub-agent sees its full prior conversation and retains all REPL state (variables, imports, computed data) from previous tasks. This makes sub-agents useful as persistent specialists:

```python
analyst = await create_agent(instructions="You are a data analyst")

t1 = await run_agent(agent_id=analyst, task="Load sales.csv and compute monthly totals")
await await_result(t1)

# The analyst still has the loaded data and computed totals in its REPL
t2 = await run_agent(agent_id=analyst, task="Now find the month-over-month growth rate")
growth = await await_result(t2)
```

The maximum recursion depth is configurable (default: 10 levels).

## Configuration

### `AgentConfig`

| Parameter | Default | Description |
|---|---|---|
| `model` | `"openai/gpt-5.2"` | Model identifier |
| `provider` | `"openrouter"` | LLM provider (determines API key and endpoint) |
| `sandbox_image` | `None` | Docker image tag; `None` for local mode |
| `code_timeout` | `3600.0` | Code execution timeout in seconds |
| `max_tool_rounds` | `50` | Max LLM-tool iterations per turn |
| `max_sub_agent_depth` | `10` | Max recursive sub-agent depth |
| `output_limit_lines` | `2000` | Truncate tool output beyond this many lines |
| `output_limit_bytes` | `50000` | Truncate tool output beyond this many bytes |
| `temperature` | `None` | LLM sampling temperature |
| `system_prompt` | `None` | Override the default system prompt (format string with `{functions_json}`, `{workspace_path}`, `{spool_path}` placeholders) |
| `get_api_key` | `None` | Custom async key resolver `(provider: str) -> str`; default reads env vars |
| `sandbox_binds` | `{}` | Host-to-container directory mounts (Docker mode) |

### Environment Variables

API keys are resolved from provider-specific environment variables:

| Provider | Environment Variable |
|---|---|
| `openrouter` | `OPENROUTER_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `google` | `GEMINI_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `xai` | `XAI_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |

A `.env` file in the current directory is loaded automatically.

For the bundled contrib tools (`internet_search`, `internet_extract`), set `PARALLEL_API_KEY`.

### CLI Flags

```
arcgeneral [message] [options]

positional:
  message                 User message (omit for interactive session)

options:
  --model MODEL           Model identifier (default: openai/gpt-5.2)
  --provider PROVIDER     LLM provider (default: openrouter)
  --image IMAGE           Docker image tag for sandbox (omit for local mode)
  --timeout SECONDS       Code execution timeout (default: 3600)
  --max-rounds N          Max tool loop iterations (default: 50)
  --functions PATH        Directory, .py file, or dotted module name (comma-separated)
  --workspace DIR         Working directory shared with agents (default: cwd)
  --context FILE          JSON file with conversation history to prepend
  --json                  Output result as JSON object
  --verbose               Enable debug logging
  --env-file PATH         Path to .env file (default: .env)
  --log-file PATH         Log file path (default: ~/Downloads/arcgeneral.log)
```

#### `--context FILE`

Prepends conversation history after the system prompt. The file must contain a JSON array of messages:

```json
[
  {"role": "user", "content": "I'm analyzing sales data"},
  {"role": "assistant", "content": "I see the file has 10k rows with date, product, and revenue columns."}
]
```

Only `"user"` and `"assistant"` roles are allowed. This is useful for context bridging from an outer harness — pass a filtered conversation history so the agent understands what's been discussed.

#### `--json`

Wraps the result in a JSON object for programmatic consumption. Only valid with a message argument (not interactive mode).

```json
// Success
{"result": "The answer is 42", "error": null}

// Failure
{"result": null, "error": "API key not found for provider 'openai'"}
```

Exactly one of `result` or `error` is non-null.

## Bundled Tools

The `contrib/` directory includes two pre-built host functions that use the [Parallel API](https://www.parallel.ai/):

- **`internet_search`** — Search the web and return relevant excerpts with source URLs.
- **`internet_extract`** — Fetch a web page or PDF and return its content as markdown.

Both require `PARALLEL_API_KEY` in the environment and the `contrib` extra (`pip install arcgeneral[contrib]`).

```bash
arcgeneral --functions ./contrib "search for recent papers on transformer efficiency"
```

## How It Works
The **core** gives an LLM a persistent IPython sandbox with host function injection. You make your own LLM calls, extract code from the response, and run it via `sandbox.execute(code, timeout)`. State persists across calls. You control the LLM provider and message format.

The **harness** (`Session` + `AgentRuntime`) adds a managed agent mode on top. It registers `create_agent`/`run_agent`/`await_result` as host functions to enable recursive sub-agent spawning, and drives the LLM loop: the LLM emits a `python` tool call, the harness executes it in the REPL, returns the output, and repeats until the LLM responds with text. Everything — web requests, file I/O, data analysis, sub-agent orchestration — is code the LLM writes and runs through that single tool. The harness also handles:
- **Message compression.** Previous turns are compressed to just the user message and final assistant response. The current turn retains full tool call detail. The complete uncompressed history is available inside the REPL as `_conversation_history`.
- **Cancellation.** Cancelling a turn (via `asyncio.CancelledError` or the CLI's Ctrl-C) rolls back the message history to the last consistent checkpoint. Sub-agent tasks are cancelled transitively.

### CLI-specific behavior

- **Ctrl-C** cancels the active turn and returns to the prompt.
- **Double Ctrl-C** exits the session.
- **`--json` mode** wraps the final response in `{"result": ..., "error": ...}` for programmatic consumption.

## LLM Client
The default client uses [OpenRouter](https://openrouter.ai/), which provides access to models from OpenAI, Anthropic, Google, and others through a single API. It manages a persistent HTTP/2 connection pool internally and cleans it up when the runtime exits.

To use a different provider, implement the `LLMClient` protocol and inject it:
```python
import httpx
from arcgeneral import AgentRuntime, AgentConfig, HostFunctionRegistry
from arcgeneral import LLMClient, CompletionResponse, CompletionChoice, CompletionMessage, TokenUsage

class AnthropicDirectClient:
    """Example: call Anthropic's API directly instead of through OpenRouter."""

    def __init__(self):
        self._http = httpx.AsyncClient(base_url="https://api.anthropic.com")

    async def complete(self, messages, *, api_key, **kwargs) -> CompletionResponse:
        resp = await self._http.post("/v1/messages", headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }, json={"model": kwargs.get("model", "claude-sonnet-4-5-20250514"), "messages": messages, "max_tokens": 4096})
        data = resp.json()
        return CompletionResponse(
            model=data["model"],
            choices=[CompletionChoice(
                message=CompletionMessage(content=data["content"][0]["text"], tool_calls=None),
                finish_reason=data["stop_reason"],
            )],
            usage=TokenUsage(
                prompt_tokens=data["usage"]["input_tokens"],
                completion_tokens=data["usage"]["output_tokens"],
            ),
        )

    async def close(self) -> None:
        """Release the HTTP connection pool."""
        await self._http.aclose()


async def main():
    client = AnthropicDirectClient()
    config = AgentConfig(model="claude-sonnet-4-5-20250514", provider="anthropic")
    runtime = AgentRuntime(config, HostFunctionRegistry(), llm_client=client)

    try:
        async with runtime:
            session = await runtime.create_session("s1")
            result = await session.run_single("What is 2 + 2?")
            print(result)
            await runtime.close_session("s1")
    finally:
        # You injected the client, so you close it.
        # The runtime only auto-closes clients it created itself.
        await client.close()
```

The `close()` method releases whatever resources your client holds — typically an HTTP connection pool. When the runtime creates its own default `OpenRouterClient`, it handles closing automatically on exit. When you inject a client, you own its lifecycle.

## Development

```bash
# Clone and install
git clone <repo-url>
cd arcgeneral
uv sync

# Run tests (requires Docker for full suite)
uv run python tests/test_e2e.py

# Build sandbox image (for Docker mode tests)
python -c "from arcgeneral.ipybox.build import build; from pathlib import Path; build('arcgeneral:sandbox', Path('sandbox-deps.txt'))"
```

## License

MIT
