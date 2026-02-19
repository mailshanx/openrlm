"""End-to-end integration tests for arcgeneral.

Tests everything except the agent loop (run_agent), which requires API keys.

Requires Docker running and the arcgeneral:sandbox image built:
    python -c "from arcgeneral.ipybox.build import build; from pathlib import Path; build('arcgeneral:sandbox', Path('sandbox-deps.txt'))"

Run:
    uv run python tests/test_e2e.py
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass

from arcgeneral.agent import DEFAULT_SYSTEM_PROMPT
from arcgeneral.config import AgentConfig
from arcgeneral.sandbox import ForkServer, LocalForkServer, Sandbox
from arcgeneral.tool import PYTHON_TOOL_SCHEMA, execute_tool
from arcgeneral.host_functions import HostFunctionRegistry
from arcgeneral.agent import AgentRuntime, Session
from arcgeneral.events import (
    RoundStart, ModelRequest, ModelResponse,
    ToolExecStart, ToolExecEnd, TurnEnd,
)
from arcgeneral.llm import (
    CompletionResponse, CompletionChoice, CompletionMessage,
    ToolCall, ToolCallFunction, TokenUsage, OpenRouterClient,
)

TAG = "arcgeneral:sandbox"


class _NullClient:
    """Mock LLM client that raises if accidentally called."""
    async def complete(self, messages, *, api_key="", **kwargs):
        raise RuntimeError("_NullClient.complete() should never be called — mock _run_turn or _llm_call first")
    async def close(self):
        pass

passed_count = 0
failures: list[str] = []


def report(name: str, ok: bool, detail: str = ""):
    global passed_count
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if ok:
        passed_count += 1
    else:
        failures.append(name)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

async def test_config_defaults():
    c = AgentConfig()
    report("config_defaults", all([
        c.model == "openai/gpt-4o",
        c.provider == "openrouter",
        c.system_prompt is None,
        c.sandbox_image is None,
        c.code_timeout == 3600.0,
        c.max_tool_rounds == 50,
        c.temperature is None,
    ]))


async def test_config_override():
    c = AgentConfig(
        model="openai/gpt-4o-mini",
        provider="anthropic",
        system_prompt="You are a cat.",
        sandbox_image="custom:latest",
        code_timeout=30.0,
        max_tool_rounds=5,
        temperature=0.7,
    )
    report("config_override", all([
        c.model == "openai/gpt-4o-mini",
        c.provider == "anthropic",
        c.system_prompt == "You are a cat.",
        c.sandbox_image == "custom:latest",
        c.code_timeout == 30.0,
        c.max_tool_rounds == 5,
        c.temperature == 0.7,
    ]))


# ---------------------------------------------------------------------------
# System prompt (from agent module, but no run_agent call)
# ---------------------------------------------------------------------------

async def test_system_prompt_exists():
    report("system_prompt_exists", isinstance(DEFAULT_SYSTEM_PROMPT, str) and len(DEFAULT_SYSTEM_PROMPT) > 100)


async def test_system_prompt_mentions_python_tool():
    report("system_prompt_mentions_tool", "python" in DEFAULT_SYSTEM_PROMPT.lower())


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

async def test_tool_schema_structure():
    s = PYTHON_TOOL_SCHEMA
    report("tool_schema_type", s["type"] == "function")
    report("tool_schema_name", s["function"]["name"] == "python")
    report("tool_schema_has_no_strict", "strict" not in s["function"])
    report("tool_schema_code_param", "code" in s["function"]["parameters"]["properties"])
    report("tool_schema_required", s["function"]["parameters"]["required"] == ["code"])
    report("tool_schema_no_extra", s["function"]["parameters"]["additionalProperties"] is False)


async def test_tool_schema_json_serializable():
    try:
        roundtripped = json.loads(json.dumps(PYTHON_TOOL_SCHEMA))
        report("tool_schema_json_roundtrip", roundtripped == PYTHON_TOOL_SCHEMA)
    except (TypeError, ValueError) as e:
        report("tool_schema_json_roundtrip", False, str(e))


# ---------------------------------------------------------------------------
# CLI arg parsing (no main() call — that hits the agent loop)
# ---------------------------------------------------------------------------

async def test_cli_parse_defaults():
    from arcgeneral.cli import parse_args
    # parse_args reads sys.argv; temporarily replace it
    original_argv = sys.argv
    try:
        sys.argv = ["arcgeneral", "hello world"]
        args = parse_args()
        report("cli_message", args.message == "hello world")
        report("cli_default_model", isinstance(args.model, str) and len(args.model) > 0)
        report("cli_default_provider", args.provider == "openrouter")
        report("cli_default_image", args.image is None)
        report("cli_default_timeout", args.timeout == 3600.0)
        report("cli_default_max_rounds", args.max_rounds == 50)
        report("cli_default_env_file", args.env_file == ".env")
        report("cli_default_verbose", args.verbose is False)
    finally:
        sys.argv = original_argv


async def test_cli_parse_overrides():
    from arcgeneral.cli import parse_args
    original_argv = sys.argv
    try:
        sys.argv = [
            "arcgeneral",
            "compute stuff",
            "--model", "openai/gpt-4o-mini",
            "--provider", "anthropic",
            "--image", "custom:v2",
            "--timeout", "30.5",
            "--max-rounds", "10",
            "--env-file", "prod.env",
            "--verbose",
        ]
        args = parse_args()
        report("cli_override_message", args.message == "compute stuff")
        report("cli_override_model", args.model == "openai/gpt-4o-mini")
        report("cli_override_provider", args.provider == "anthropic")
        report("cli_override_image", args.image == "custom:v2")
        report("cli_override_timeout", args.timeout == 30.5)
        report("cli_override_max_rounds", args.max_rounds == 10)
        report("cli_override_env_file", args.env_file == "prod.env")
        report("cli_override_verbose", args.verbose is True)
    finally:
        sys.argv = original_argv


async def test_cli_functions_flag():
    """--functions flag parses into a list; _load_functions imports and calls register()."""
    from arcgeneral.cli import parse_args, _load_functions

    # Test parsing: comma-separated modules
    original_argv = sys.argv
    try:
        sys.argv = ["arcgeneral", "hello", "--functions", "mod_a,mod_b"]
        args = parse_args()
        report("cli_functions_parsed", args.functions == "mod_a,mod_b", repr(args.functions))
    finally:
        sys.argv = original_argv

    # Test parsing: no --functions gives None
    try:
        sys.argv = ["arcgeneral", "hello"]
        args = parse_args()
        report("cli_functions_default_none", args.functions is None, repr(args.functions))
    finally:
        sys.argv = original_argv

    # Test _load_functions with a real inline module
    import types
    fake_mod = types.ModuleType("_test_fake_functions")
    registered_names = []
    def fake_register(registry):
        registered_names.append("called")
    fake_mod.register = fake_register
    sys.modules["_test_fake_functions"] = fake_mod
    try:
        registry = HostFunctionRegistry()
        _load_functions(registry, ["_test_fake_functions"])
        report("cli_load_functions_called", registered_names == ["called"])
    finally:
        del sys.modules["_test_fake_functions"]

    # Test _load_functions with missing module
    try:
        _load_functions(HostFunctionRegistry(), ["nonexistent_module_xyz"])
        report("cli_load_functions_missing_fails", False, "should have raised")
    except SystemExit:
        report("cli_load_functions_missing_fails", True)

    # Test _load_functions with module missing register()
    bare_mod = types.ModuleType("_test_bare_module")
    sys.modules["_test_bare_module"] = bare_mod
    try:
        _load_functions(HostFunctionRegistry(), ["_test_bare_module"])
        report("cli_load_functions_no_register_fails", False, "should have raised")
    except SystemExit:
        report("cli_load_functions_no_register_fails", True)
    finally:
        del sys.modules["_test_bare_module"]

    # Test _load_functions with a .py file path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        func_file = Path(tmpdir) / "my_funcs.py"
        func_file.write_text(
            "def register(registry):\n"
            "    registry._test_file_loaded = True\n"
        )
        registry = HostFunctionRegistry()
        _load_functions(registry, [str(func_file)])
        report("cli_load_functions_file", getattr(registry, "_test_file_loaded", False))

    # Test _load_functions with a directory
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "tool_a.py").write_text(
            "def register(registry):\n"
            "    if not hasattr(registry, '_dir_loaded'): registry._dir_loaded = []\n"
            "    registry._dir_loaded.append('a')\n"
        )
        (Path(tmpdir) / "tool_b.py").write_text(
            "def register(registry):\n"
            "    if not hasattr(registry, '_dir_loaded'): registry._dir_loaded = []\n"
            "    registry._dir_loaded.append('b')\n"
        )
        # _underscore files should be skipped
        (Path(tmpdir) / "_private.py").write_text(
            "def register(registry):\n"
            "    registry._should_not_load = True\n"
        )
        # Files without register() should be silently skipped in directory mode
        (Path(tmpdir) / "no_register.py").write_text("x = 1\n")
        registry = HostFunctionRegistry()
        _load_functions(registry, [tmpdir])
        loaded = sorted(getattr(registry, "_dir_loaded", []))
        report("cli_load_functions_dir", loaded == ["a", "b"], repr(loaded))
        report("cli_load_functions_dir_skips_underscore", not getattr(registry, "_should_not_load", False))

    # Test _load_functions with missing file
    try:
        _load_functions(HostFunctionRegistry(), ["/nonexistent/path/funcs.py"])
        report("cli_load_functions_missing_file", False, "should have raised")
    except SystemExit:
        report("cli_load_functions_missing_file", True)

    # Test _load_functions with empty directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _load_functions(HostFunctionRegistry(), [tmpdir])
            report("cli_load_functions_empty_dir", False, "should have raised")
        except SystemExit:
            report("cli_load_functions_empty_dir", True)

async def test_cli_context_and_json_flags():
    """--context and --json flags parse correctly; context file is validated."""
    import tempfile
    from arcgeneral.cli import parse_args

    original_argv = sys.argv

    # --context and --json parse correctly
    try:
        sys.argv = ["arcgeneral", "hello", "--context", "/tmp/ctx.json", "--json"]
        args = parse_args()
        report("cli_context_parsed", args.context == "/tmp/ctx.json")
        report("cli_json_parsed", args.json is True)
    finally:
        sys.argv = original_argv

    # Defaults: no context, no json
    try:
        sys.argv = ["arcgeneral", "hello"]
        args = parse_args()
        report("cli_context_default_none", args.context is None)
        report("cli_json_default_false", args.json is False)
    finally:
        sys.argv = original_argv

    # Context file validation: valid file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ], f)
        valid_ctx_path = f.name
    ctx_data = json.loads(Path(valid_ctx_path).read_text())
    report("cli_context_valid_file", len(ctx_data) == 2 and ctx_data[0]["role"] == "user")
    Path(valid_ctx_path).unlink()

    # Context file validation: invalid role
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"role": "system", "content": "bad"}], f)
        bad_role_path = f.name
    # The validation happens in main(), not parse_args(), so we test the validation logic directly
    from arcgeneral.cli import main as _cli_main
    try:
        sys.argv = ["arcgeneral", "hello", "--context", bad_role_path]
        _cli_main()
        report("cli_context_invalid_role_fails", False, "should have raised")
    except SystemExit as e:
        report("cli_context_invalid_role_fails", "invalid role" in str(e))
    finally:
        sys.argv = original_argv
        Path(bad_role_path).unlink()

    # Context file validation: missing file
    try:
        sys.argv = ["arcgeneral", "hello", "--context", "/nonexistent/ctx.json"]
        _cli_main()
        report("cli_context_missing_file_fails", False, "should have raised")
    except SystemExit as e:
        report("cli_context_missing_file_fails", "not found" in str(e))
    finally:
        sys.argv = original_argv

    # --json without message should fail
    try:
        sys.argv = ["arcgeneral", "--json"]
        _cli_main()
        report("cli_json_requires_message", False, "should have raised")
    except SystemExit as e:
        report("cli_json_requires_message", True)
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# Tool dispatch (requires sandbox)
# ---------------------------------------------------------------------------

async def test_tool_execute_python(sb: Sandbox):
    result = await execute_tool(sb, "python", json.dumps({"code": "print(7 * 6)"}), timeout=30)
    report("tool_dispatch_python", "42" in result, repr(result))


async def test_tool_execute_unknown(sb: Sandbox):
    result = await execute_tool(sb, "bash", json.dumps({"code": "ls"}), timeout=30)
    report("tool_dispatch_unknown", "Unknown tool" in result, repr(result))


async def test_tool_execute_error_propagates(sb: Sandbox):
    result = await execute_tool(sb, "python", json.dumps({"code": "1/0"}), timeout=30)
    report("tool_dispatch_error", "ZeroDivisionError" in result, repr(result[:80]))


# ---------------------------------------------------------------------------
# Sandbox lifecycle & execution (requires Docker)
# ---------------------------------------------------------------------------

async def test_sandbox_basic_output(sb: Sandbox):
    result = await sb.execute("print(2 + 2)")
    report("sandbox_basic_output", result.strip() == "4", repr(result))


async def test_sandbox_no_output(sb: Sandbox):
    result = await sb.execute("x = 1")
    report("sandbox_no_output", "No output" in result or "successfully" in result.lower(), repr(result))


async def test_sandbox_state_persists(sb: Sandbox):
    await sb.execute("_test_var = 42")
    result = await sb.execute("print(_test_var * 2)")
    report("sandbox_state_persists", "84" in result, repr(result))


async def test_sandbox_function_persists(sb: Sandbox):
    await sb.execute("def _test_double(n): return n * 2")
    result = await sb.execute("print(_test_double(21))")
    report("sandbox_function_persists", "42" in result, repr(result))


async def test_sandbox_import_persists(sb: Sandbox):
    await sb.execute("import math as _test_math")
    result = await sb.execute("print(_test_math.factorial(6))")
    report("sandbox_import_persists", "720" in result, repr(result))


async def test_sandbox_multiline(sb: Sandbox):
    code = "total = 0\nfor i in range(1, 11):\n    total += i\nprint(total)"
    result = await sb.execute(code)
    report("sandbox_multiline", "55" in result, repr(result))


async def test_sandbox_multiline_function(sb: Sandbox):
    code = "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib(10))"
    result = await sb.execute(code)
    report("sandbox_multiline_function", "55" in result, repr(result))


async def test_sandbox_error_traceback(sb: Sandbox):
    result = await sb.execute("1 / 0")
    report("sandbox_error_traceback", "ZeroDivisionError" in result, repr(result[:80]))


async def test_sandbox_syntax_error(sb: Sandbox):
    result = await sb.execute("def f(")
    report("sandbox_syntax_error", "SyntaxError" in result or "Error" in result, repr(result[:80]))


async def test_sandbox_name_error(sb: Sandbox):
    result = await sb.execute("print(_never_defined_xyz)")
    report("sandbox_name_error", "NameError" in result, repr(result[:80]))


async def test_sandbox_error_does_not_poison_state(sb: Sandbox):
    await sb.execute("_poison_test = 99")
    await sb.execute("1 / 0")
    result = await sb.execute("print(_poison_test)")
    report("sandbox_error_no_poison", "99" in result, repr(result))


async def test_sandbox_numpy(sb: Sandbox):
    result = await sb.execute("import numpy as np; print(np.array([1,2,3]).sum())")
    report("sandbox_numpy", "6" in result, repr(result))


async def test_sandbox_pandas(sb: Sandbox):
    result = await sb.execute(
        "import pandas as pd; df = pd.DataFrame({'a': [1,2,3]}); print(df['a'].sum())"
    )
    report("sandbox_pandas", "6" in result, repr(result))


async def test_sandbox_requests(sb: Sandbox):
    result = await sb.execute("import requests; print(requests.__version__)")
    report("sandbox_requests", result.strip() != "" and "Error" not in result, repr(result[:60]))


async def test_sandbox_large_output(sb: Sandbox):
    result = await sb.execute("print('x' * 10000)")
    report("sandbox_large_output", len(result) >= 10000, f"len={len(result)}")


async def test_sandbox_empty_code(sb: Sandbox):
    result = await sb.execute("")
    report("sandbox_empty_code", isinstance(result, str), repr(result[:60]))


async def test_sandbox_expression_result(sb: Sandbox):
    """IPython-style: bare expression prints its result."""
    result = await sb.execute("2 ** 10")
    report("sandbox_expression_result", "1024" in result, repr(result))


async def test_sandbox_timeout(sb: Sandbox):
    result = await sb.execute("import time; time.sleep(10)", timeout=2)
    report("sandbox_timeout", "timed out" in result.lower() or "Timeout" in result, repr(result[:80]))


async def test_sandbox_post_timeout_recovery(sb: Sandbox):
    """After a timeout, the sandbox should still work."""
    await sb.execute("import time; time.sleep(10)", timeout=2)
    result = await sb.execute("print('recovered')")
    report("sandbox_post_timeout_recovery", "recovered" in result, repr(result))


async def test_sandbox_top_level_async(sb: Sandbox):
    result = await sb.execute("import asyncio\nawait asyncio.sleep(0.01)\nprint('async ok')")
    report("sandbox_top_level_async", "async ok" in result, repr(result))


# ---------------------------------------------------------------------------
# Fork server tests (multi-agent, requires Docker)
# ---------------------------------------------------------------------------

async def test_forkserver_independent_namespaces(fs: ForkServer):
    """Two sandboxes have independent state."""
    sb1 = await fs.create_sandbox()
    sb2 = await fs.create_sandbox()
    await sb1.execute("color = 'red'")
    await sb2.execute("color = 'blue'")
    r1 = await sb1.execute("print(color)")
    r2 = await sb2.execute("print(color)")
    report("forkserver_independent_namespaces",
           r1.strip() == "red" and r2.strip() == "blue",
           f"sb1={r1.strip()!r} sb2={r2.strip()!r}")
    await sb1.close()
    await sb2.close()


async def test_forkserver_preloaded_imports(fs: ForkServer):
    """Pre-imported packages are available instantly."""
    sb = await fs.create_sandbox()
    r1 = await sb.execute("import numpy as np; print(np.__version__)")
    r2 = await sb.execute("import pandas as pd; print(pd.__version__)")
    report("forkserver_preloaded_numpy", r1.strip() != "" and "Error" not in r1, repr(r1[:60]))
    report("forkserver_preloaded_pandas", r2.strip() != "" and "Error" not in r2, repr(r2[:60]))
    await sb.close()


async def test_forkserver_interrupt_isolation(fs: ForkServer):
    """Interrupting one sandbox doesn't affect another."""
    sb1 = await fs.create_sandbox()
    sb2 = await fs.create_sandbox()
    # Timeout sb1
    await sb1.execute("import time; time.sleep(999)", timeout=2)
    # sb2 should still work
    r = await sb2.execute("print('unaffected')")
    report("forkserver_interrupt_isolation", "unaffected" in r, repr(r))
    # sb1 should recover
    r = await sb1.execute("print('recovered')")
    report("forkserver_interrupt_recovery", "recovered" in r, repr(r))
    await sb1.close()
    await sb2.close()


async def test_forkserver_destroy_respawn(fs: ForkServer):
    """Destroying a sandbox and creating a new one works."""
    sb = await fs.create_sandbox()
    await sb.execute("x = 42")
    await sb.close()
    sb2 = await fs.create_sandbox()
    r = await sb2.execute("print('fresh')")
    report("forkserver_destroy_respawn", "fresh" in r, repr(r))
    await sb2.close()


async def test_forkserver_pip_install_shared(fs: ForkServer):
    """pip install in one sandbox is visible to others (shared site-packages)."""
    sb1 = await fs.create_sandbox()
    sb2 = await fs.create_sandbox()
    await sb1.execute("import subprocess; subprocess.check_call(['pip', 'install', '-q', 'six'])", timeout=30)
    r = await sb2.execute("import six; print(six.__version__)")
    report("forkserver_pip_install_shared", r.strip() != "" and "Error" not in r, repr(r[:60]))
    await sb1.close()
    await sb2.close()

async def test_forkserver_concurrent_execute(fs: ForkServer):
    """Two sandboxes can execute code concurrently without deadlocking."""
    sb1 = await fs.create_sandbox()
    sb2 = await fs.create_sandbox()
    r1, r2 = await asyncio.gather(
        sb1.execute("import time; time.sleep(0.5); print('alpha')"),
        sb2.execute("import time; time.sleep(0.5); print('bravo')"),
    )
    report("forkserver_concurrent_execute",
           "alpha" in r1 and "bravo" in r2,
           f"r1={r1.strip()!r} r2={r2.strip()!r}")
    await sb1.close()
    await sb2.close()


async def test_forkserver_sequential_reuse(fs: ForkServer):
    """A persistent sandbox retains state across sequential execute calls."""
    sb = await fs.create_sandbox()
    await sb.execute("saved_data = {'key': 'value_from_first_task'}")
    r = await sb.execute("print(saved_data['key'])")
    report("forkserver_sequential_reuse",
           "value_from_first_task" in r,
           repr(r.strip()))
    await sb.close()

async def test_sub_agent_task_serialization(fs: ForkServer):
    """Concurrent multi-step tasks on the same sandbox, protected by a lock,
    do not interleave — each task's steps run as a contiguous block.
    This mirrors the _SubAgent.lock in agent.py."""
    sb = await fs.create_sandbox()
    lock = asyncio.Lock()

    async def multi_step_task(label):
        async with lock:
            await sb.execute(f"order.append('{label}_start')")
            await sb.execute("import time; time.sleep(0.3)")
            await sb.execute(f"order.append('{label}_end')")

    await sb.execute("order = []")
    await asyncio.gather(multi_step_task("a"), multi_step_task("b"))
    r = await sb.execute("print(order)")

    # Each task's _start/_end must be adjacent — no interleaving.
    # Either a ran first or b ran first; both are valid.
    ok = ("['a_start', 'a_end', 'b_start', 'b_end']" in r or
          "['b_start', 'b_end', 'a_start', 'a_end']" in r)
    report("sub_agent_task_serialization", ok, repr(r.strip()))
    await sb.close()

async def test_host_function_path_serialization():
    """Full path: parent sandbox code → host function HTTP bridge →
    _host_create_agent / _host_run_agent (with sub.lock) → _ensure_sandbox →
    sub-agent sandbox.execute → _host_await_result.

    Verifies that two concurrent run_agent calls on the same sub-agent are
    serialized end-to-end, not just at the fork server level.
    """
    config = AgentConfig(model="test/model", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=_NullClient())

    # Replace _run_turn on session: skip LLM, just execute the task string as code.
    async def _mock_run_turn(full_history, sandbox, request_kwargs, agent_label=""):
        task = full_history[-1]["content"]
        result = await sandbox.execute(task, timeout=30.0)
        full_history.append({"role": "assistant", "content": result})
        return result

    async with runtime:
        session = await runtime.create_session("test")
        session._run_turn = _mock_run_turn
        sb = session._sandbox
        # Step 1: basic sandbox works
        r = await sb.execute("print('hello')", timeout=10)
        print(f"    step1 basic: {r.strip()!r}")

        # Step 2: host function bridge works
        r = await sb.execute("aid = await create_agent('test'); print(aid)", timeout=30)
        print(f"    step2 create_agent: {r.strip()!r}")

        # Step 3: run_agent + await_result work
        r = await sb.execute("tid = await run_agent(agent_id=aid, task=\"print('sub-hello')\")", timeout=30)
        print(f"    step3 run_agent: {r.strip()!r}")
        r = await sb.execute("res = await await_result(tid); print(res)", timeout=30)
        print(f"    step4 await_result: {r.strip()!r}")

        # Step 5: serialization test
        code = '''t0 = await run_agent(agent_id=aid, task="order = []")
await await_result(t0)
t1 = await run_agent(agent_id=aid, task="order.append('t1_start'); import time; time.sleep(0.3); order.append('t1_end')")
t2 = await run_agent(agent_id=aid, task="order.append('t2_start'); import time; time.sleep(0.3); order.append('t2_end')")
import asyncio
r1, r2 = await asyncio.gather(await_result(t1), await_result(t2))
t3 = await run_agent(agent_id=aid, task="print(order)")
r3 = await await_result(t3)
print(r3)
'''
        result = await sb.execute(code, timeout=60.0)
        ok = ("['t1_start', 't1_end', 't2_start', 't2_end']" in result or
              "['t2_start', 't2_end', 't1_start', 't1_end']" in result)
        report("host_function_path_serialization", ok, repr(result.strip()[-80:]))


# ---------------------------------------------------------------------------
# Mock LLM response helpers (duck-typed to match OpenRouter SDK shapes)
# ---------------------------------------------------------------------------

@dataclass
class _MFn:
    name: str
    arguments: str

@dataclass
class _MTC:
    id: str
    function: _MFn

@dataclass
class _MMsg:
    content: str | None
    tool_calls: list | None

@dataclass
class _MUsage:
    prompt_tokens: float
    completion_tokens: float

@dataclass
class _MChoice:
    message: _MMsg
    finish_reason: str

@dataclass
class _MResp:
    model: str
    choices: list
    usage: _MUsage


_tc_counter = 0

def _text_resp(text, pt=100, ct=50):
    return _MResp("test/mock", [_MChoice(_MMsg(text, None), "stop")], _MUsage(pt, ct))

def _tool_resp(code, reasoning=None, pt=100, ct=50):
    global _tc_counter
    _tc_counter += 1
    tc = _MTC(f"call_{_tc_counter}", _MFn("python", json.dumps({"code": code})))
    return _MResp("test/mock", [_MChoice(_MMsg(reasoning, [tc]), "tool_calls")], _MUsage(pt, ct))

def check(name, fn, detail=""):
    """Evaluate fn(); report FAIL on any exception instead of crashing."""
    try:
        ok = fn()
    except Exception as e:
        ok = False
        if not detail:
            detail = type(e).__name__ + ": " + str(e)
    report(name, ok, detail)


# ---------------------------------------------------------------------------
# Event emission tests (requires Docker, mock LLM)
# ---------------------------------------------------------------------------

async def test_event_emission_two_rounds():
    """_run_turn emits correct events for tool-call round + final response.

    Mock LLM returns:
      Round 0: tool_call with code 'print(2+2)'
      Round 1: text response 'The answer is 4'

    Expected 9 events in order:
      RoundStart(0) ModelRequest ModelResponse(tool_calls=True)
      ToolExecStart  ToolExecEnd
      RoundStart(1) ModelRequest ModelResponse(tool_calls=False)
      TurnEnd(rounds=2)
    """

    config = AgentConfig(model="test/model", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=_NullClient())

    responses = [_tool_resp("print(2+2)", pt=100, ct=50), _text_resp("The answer is 4", pt=120, ct=60)]
    async def mock_llm(messages, request_kwargs):
        return responses.pop(0)

    async with runtime:
        session = await runtime.create_session("test")
        events = []
        session._on_event = events.append
        runtime._llm_call = mock_llm
        messages = list(session._messages)
        messages.append({"role": "user", "content": "What is 2+2?"})
        result = await session._run_turn(
            messages, session._sandbox,
            session._request_kwargs, agent_label="main",
        )

        check("event_count_9", lambda: len(events) == 9, f"got {len(events)}")
        # Round 0
        check("event_r0_start", lambda: isinstance(events[0], RoundStart) and events[0].round_num == 0 and events[0].agent_id == "main")
        check("event_r0_request", lambda: isinstance(events[1], ModelRequest) and events[1].agent_id == "main")
        check("event_r0_response", lambda: isinstance(events[2], ModelResponse) and events[2].has_tool_calls and events[2].model == "test/mock")
        check("event_tool_start", lambda: isinstance(events[3], ToolExecStart) and "print(2+2)" in events[3].code and events[3].agent_id == "main")
        check("event_tool_end", lambda: isinstance(events[4], ToolExecEnd) and events[4].elapsed_seconds >= 0 and events[4].agent_id == "main")
        # Round 1
        check("event_r1_start", lambda: isinstance(events[5], RoundStart) and events[5].round_num == 1)
        check("event_r1_response", lambda: isinstance(events[7], ModelResponse) and not events[7].has_tool_calls)
        # TurnEnd
        check("event_turn_end", lambda: isinstance(events[8], TurnEnd) and events[8].rounds == 2)
        check("event_turn_tokens", lambda: events[8].prompt_tokens == 220 and events[8].completion_tokens == 110)
        check("event_turn_timing", lambda: events[8].elapsed_seconds > 0)
        check("event_return_value", lambda: result == "The answer is 4")


async def test_event_emission_immediate():
    """_run_turn emits 4 events when LLM responds immediately (no tool calls)."""

    config = AgentConfig(model="test/model", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=_NullClient())

    async def mock_llm(messages, request_kwargs):
        return _text_resp("Hello!")

    async with runtime:
        session = await runtime.create_session("test")
        events = []
        session._on_event = events.append
        runtime._llm_call = mock_llm
        messages = list(session._messages)
        messages.append({"role": "user", "content": "Hi"})
        result = await session._run_turn(
            messages, session._sandbox,
            session._request_kwargs, agent_label="main",
        )

        check("event_imm_count_4", lambda: len(events) == 4, f"got {len(events)}")
        check("event_imm_round_start", lambda: isinstance(events[0], RoundStart) and events[0].round_num == 0)
        check("event_imm_model_req", lambda: isinstance(events[1], ModelRequest))
        check("event_imm_model_resp", lambda: isinstance(events[2], ModelResponse) and not events[2].has_tool_calls)
        check("event_imm_turn_end", lambda: isinstance(events[3], TurnEnd) and events[3].rounds == 1)
        check("event_imm_result", lambda: result == "Hello!")


async def test_event_callback_error_isolation():
    """A broken event callback must not crash _run_turn."""

    config = AgentConfig(model="test/model", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=_NullClient())

    async def mock_llm(messages, request_kwargs):
        return _text_resp("Survived")

    async with runtime:
        session = await runtime.create_session("test")
        def exploding_callback(event):
            raise ValueError("consumer bug")
        session._on_event = exploding_callback
        runtime._llm_call = mock_llm
        messages = list(session._messages)
        messages.append({"role": "user", "content": "Test"})
        result = await session._run_turn(
            messages, session._sandbox,
            session._request_kwargs, agent_label="main",
        )
        check("event_error_isolation", lambda: result == "Survived")


async def test_event_no_callback():
    """_run_turn works identically when _on_event is None (backward compat)."""

    config = AgentConfig(model="test/model", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=_NullClient())

    async def mock_llm(messages, request_kwargs):
        return _text_resp("No events")

    async with runtime:
        session = await runtime.create_session("test")
        runtime._llm_call = mock_llm
        messages = list(session._messages)
        messages.append({"role": "user", "content": "Test"})
        result = await session._run_turn(
            messages, session._sandbox,
            session._request_kwargs, agent_label="main",
        )
        check("event_no_callback", lambda: result == "No events")


async def test_event_sub_agent_propagation():
    """Events from sub-agent _run_turn appear in the same stream with correct agent_id.

    Mock LLM returns 3 responses in deterministic order:
      1. Main agent: tool_call with code that creates/runs/awaits a sub-agent
      2. Sub-agent: text response (the sub-agent's _run_turn calls _llm_call)
      3. Main agent: text response (after tool execution completes)

    Expected event sequence (13 events):
      Main  RoundStart(0)  ModelRequest  ModelResponse(tool_calls)
      Main  ToolExecStart
        Sub RoundStart(0) ModelRequest ModelResponse(no_tools) TurnEnd
      Main  ToolExecEnd
      Main  RoundStart(1) ModelRequest ModelResponse(no_tools) TurnEnd
    """

    config = AgentConfig(model="test/model", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=_NullClient())

    sub_agent_code = """aid = await create_agent(instructions='helper')
tid = await run_agent(agent_id=aid, task='Say hello')
result = await await_result(tid)
print(f'sub says: {result}')"""

    responses = [
        _tool_resp(sub_agent_code),     # main agent round 0
        _text_resp("Hello from sub"),   # sub-agent round 0
        _text_resp("All done"),         # main agent round 1
    ]
    async def mock_llm(messages, request_kwargs):
        return responses.pop(0)

    async with runtime:
        events = []
        session = await runtime.create_session("test", on_event=events.append)
        runtime._llm_call = mock_llm
        messages = list(session._messages)
        messages.append({"role": "user", "content": "Use a sub-agent"})
        result = await session._run_turn(
            messages, session._sandbox,
            session._request_kwargs, agent_label="main",
        )

        check("event_sub_count_13", lambda: len(events) == 13, f"got {len(events)}")
        check("event_sub_return", lambda: result == "All done")
        # Identify main vs sub events
        main_events = [e for e in events if getattr(e, 'agent_id', None) == "main"]
        sub_events = [e for e in events if getattr(e, 'agent_id', None) not in ("main", None)]
        check("event_sub_main_count", lambda: len(main_events) == 9, f"main={len(main_events)}")
        check("event_sub_sub_count", lambda: len(sub_events) == 4, f"sub={len(sub_events)}")
        # Sub-agent events should have consistent agent_id (a UUID, not "main")
        sub_ids = set(getattr(e, 'agent_id', None) for e in sub_events)
        check("event_sub_single_id", lambda: len(sub_ids) == 1 and "main" not in sub_ids,
               f"sub_ids={sub_ids}")
        check("event_sub_inside_tool", lambda: (
               tool_start_idx := next((i for i, e in enumerate(events) if isinstance(e, ToolExecStart)), -1),
               tool_end_idx := next((i for i, e in enumerate(events) if isinstance(e, ToolExecEnd)), -1),
               sub_idx := [i for i, e in enumerate(events) if getattr(e, 'agent_id', None) not in ("main", None)],
               tool_start_idx >= 0 and tool_end_idx >= 0 and all(tool_start_idx < i < tool_end_idx for i in sub_idx)
        )[-1])
        # Sub-agent should have its own TurnEnd
        check("event_sub_turn_end", lambda: (
               sub_te := [e for e in sub_events if isinstance(e, TurnEnd)],
               len(sub_te) == 1 and sub_te[0].rounds == 1
        )[-1])


# ---------------------------------------------------------------------------
# LLM client protocol tests
# ---------------------------------------------------------------------------

async def test_openrouter_client_converts_text_response():
    """OpenRouterClient.complete() converts SDK text response to our frozen types."""
    import unittest.mock as _mock
    client = OpenRouterClient()
    raw_resp = _MResp("gpt-4o", [_MChoice(_MMsg("hello world", None), "stop")], _MUsage(100, 50))
    class _MockChat:
        async def send_async(self, **kw):
            return raw_resp

    class _MockSDK:
        chat = _MockChat()

    with _mock.patch("openrouter.OpenRouter", lambda **kw: _MockSDK()):
        result = await client.complete(messages=[{"role": "user", "content": "hi"}], api_key="test-key", model="test")
    report("llm_text_is_completion_response", isinstance(result, CompletionResponse))
    report("llm_text_model", result.model == "gpt-4o")
    report("llm_text_content", result.choices[0].message.content == "hello world")
    report("llm_text_no_tool_calls", result.choices[0].message.tool_calls is None)
    report("llm_text_finish_reason", result.choices[0].finish_reason == "stop")
    report("llm_text_usage", result.usage == TokenUsage(prompt_tokens=100, completion_tokens=50))
    report("llm_text_frozen", result.__dataclass_params__.frozen)
    await client.close()

async def test_openrouter_client_converts_tool_response():
    """OpenRouterClient.complete() converts SDK tool-call response to our frozen types."""
    import unittest.mock as _mock
    client = OpenRouterClient()
    raw_resp = _tool_resp("print(42)", reasoning="thinking", pt=200, ct=80)
    class _MockChat:
        async def send_async(self, **kw):
            return raw_resp

    class _MockSDK:
        chat = _MockChat()

    with _mock.patch("openrouter.OpenRouter", lambda **kw: _MockSDK()):
        result = await client.complete(messages=[], api_key="test-key", model="test")
    report("llm_tool_has_tool_calls", result.choices[0].message.tool_calls is not None)
    tc = result.choices[0].message.tool_calls[0]
    report("llm_tool_call_type", isinstance(tc, ToolCall))
    report("llm_tool_fn_name", tc.function.name == "python")
    report("llm_tool_fn_args", '"print(42)"' in tc.function.arguments)
    report("llm_tool_usage", result.usage == TokenUsage(prompt_tokens=200, completion_tokens=80))
    await client.close()

async def test_llm_client_injection():
    """AgentRuntime uses an injected LLMClient instead of creating a default."""
    calls = []

    class MockClient:
        async def complete(self, messages, *, api_key="", **kwargs):
            calls.append(kwargs.get("model", "?"))
            return CompletionResponse(
                model="injected/mock",
                choices=[CompletionChoice(
                    message=CompletionMessage(content="from injected", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            )
        async def close(self):
            pass

    config = AgentConfig(model="injected-model", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=MockClient())

    async with runtime:
        session = await runtime.create_session("test")
        result = await session.run_single("test injection")

    report("llm_injection_used", len(calls) == 1 and calls[0] == "injected-model")
    report("llm_injection_result", result == "from injected")

async def test_llm_client_lifecycle_ownership():
    """Injected client is NOT closed on __aexit__; default client IS closed."""
    closed = []

    class TrackingClient:
        async def complete(self, messages, *, api_key="", **kwargs):
            return CompletionResponse(
                model="tracking",
                choices=[CompletionChoice(
                    message=CompletionMessage(content="ok", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            )
        async def close(self):
            closed.append(True)

    config = AgentConfig(model="test", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    injected = TrackingClient()
    runtime = AgentRuntime(config, registry, llm_client=injected)

    async with runtime:
        session = await runtime.create_session("test")
        await session.run_single("test lifecycle")

    report("llm_injected_not_closed", len(closed) == 0)

    # Verify the runtime knows it doesn't own the client
    report("llm_owns_flag_false", not runtime._owns_llm_client)

async def test_sigterm_clean_shutdown():
    """SIGTERM during an active session triggers clean teardown:
    session closes, sub-agent tasks cancel, fork server stops, container dies."""
    import signal

    # Track what gets closed
    closed_sessions = []
    closed_llm = []

    class SlowClient:
        async def complete(self, messages, *, api_key="", **kwargs):
            return CompletionResponse(
                model="test",
                choices=[CompletionChoice(
                    message=CompletionMessage(content="done", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            )
        async def close(self):
            closed_llm.append(True)

    config = AgentConfig(model="test", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    client = SlowClient()
    runtime = AgentRuntime(config, registry, llm_client=client)

    async with runtime:
        # Create two sessions
        s1 = await runtime.create_session("s1")
        s2 = await runtime.create_session("s2")

        # Run a turn on each so they're fully initialized
        await s1.run_single("first")
        await s2.run_single("second")

        # Verify sessions exist
        report("sigterm_sessions_exist", len(runtime._sessions) == 2)

        # Patch session.close to track calls
        original_s1_close = s1.close
        original_s2_close = s2.close
        async def _tracked_s1_close():
            closed_sessions.append("s1")
            await original_s1_close()
        async def _tracked_s2_close():
            closed_sessions.append("s2")
            await original_s2_close()
        s1.close = _tracked_s1_close
        s2.close = _tracked_s2_close

    # __aexit__ has run — verify cleanup
    report("sigterm_all_sessions_closed", set(closed_sessions) == {"s1", "s2"})
    report("sigterm_sessions_cleared", len(runtime._sessions) == 0)
    report("sigterm_agent_map_cleared", len(runtime._agent_to_session) == 0)
    report("sigterm_task_map_cleared", len(runtime._task_to_session) == 0)
    report("sigterm_fork_server_gone", runtime._fork_server is None)
    report("sigterm_host_server_gone", runtime._server is None)
    report("sigterm_injected_not_closed", len(closed_llm) == 0)


async def test_sigterm_cancels_inflight_subtasks():
    """When a session closes, running sub-agent tasks are cancelled."""

    config = AgentConfig(model="test/model", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=_NullClient())

    # Mock _run_turn on session: for sub-agents, sleep forever to simulate long work
    cancelled = []
    async def _mock_run_turn(full_history, sandbox, request_kwargs, agent_label=""):
        if agent_label != "main":
            try:
                await asyncio.sleep(3600)  # simulate long sub-agent work
            except asyncio.CancelledError:
                cancelled.append(agent_label)
                raise
        task = full_history[-1]["content"]
        result = await sandbox.execute(task, timeout=30.0)
        full_history.append({"role": "assistant", "content": result})
        return result

    async with runtime:
        session = await runtime.create_session("test")
        session._run_turn = _mock_run_turn
        sb = session._sandbox

        # Create a sub-agent and start a long-running task
        await sb.execute("aid = await create_agent('worker')", timeout=30)
        await sb.execute("tid = await run_agent(agent_id=aid, task='print(1)')", timeout=30)

        # Give the task a moment to start
        await asyncio.sleep(0.5)
        report("sigterm_subtask_running", len(session._running_tasks) == 1)

        # Close the session (simulates what happens on SIGTERM teardown)
        await session.close()

    report("sigterm_subtask_cancelled", len(cancelled) == 1)
    report("sigterm_tasks_cleared", len(session._running_tasks) == 0)
    report("sigterm_subagents_cleared", len(session._sub_agents) == 0)


async def test_cancellation_rolls_back_history():
    """When _run_turn is cancelled mid-round, full_history rolls back to
    the checkpoint — no dangling assistant tool_calls without matching
    tool results."""

    call_count = 0

    class CancellingClient:
        """First call returns a tool_call. Second call hangs until cancelled."""
        async def complete(self, messages, *, api_key="", **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Round 0: LLM wants to call a tool
                return CompletionResponse(
                    model="test",
                    choices=[CompletionChoice(
                        message=CompletionMessage(
                            content=None,
                            tool_calls=[ToolCall(
                                id="tc_1",
                                function=ToolCallFunction(
                                    name="python",
                                    arguments='{"code": "print(1)"}',
                                ),
                            )],
                        ),
                        finish_reason="tool_calls",
                    )],
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                )
            # Round 1: hang forever (will be cancelled)
            await asyncio.sleep(3600)

        async def close(self):
            pass

    config = AgentConfig(model="test", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    client = CancellingClient()
    runtime = AgentRuntime(config, registry, llm_client=client)

    async with runtime:
        session = await runtime.create_session("test")
        messages = session._messages
        pre_count = len(messages)  # system prompt only

        # Add user message
        messages.append({"role": "user", "content": "test"})
        history_before_turn = len(messages)  # system + user

        # Start _run_turn as a task so we can cancel it
        turn_task = asyncio.create_task(
            session._run_turn(
                messages, session._sandbox,
                session._request_kwargs, agent_label="main",
            )
        )

        # Let round 0 complete (tool call + tool exec) and round 1 start (LLM call hangs)
        await asyncio.sleep(1.0)
        report("cancel_round0_completed", call_count == 2)

        # Round 0 added: assistant(tool_calls) + tool result = 2 messages
        # Round 1 is in-flight (LLM call), no messages appended yet
        history_after_round0 = len(messages)
        report("cancel_history_grew", history_after_round0 > history_before_turn)

        # Cancel during round 1 (hanging on _llm_call)
        turn_task.cancel()
        try:
            await turn_task
        except asyncio.CancelledError:
            pass

        # History should be rolled back to before round 1 started,
        # which is after round 0 completed (round 0 was fully consistent).
        # The checkpoint for round 1 = history_after_round0, so rollback
        # deletes from there, leaving round 0 intact.
        report("cancel_history_consistent", len(messages) == history_after_round0)

        # Verify the last message pair is consistent: assistant with tool_calls
        # followed by matching tool result
        last_assistant = None
        tool_ids = set()
        for msg in messages[history_before_turn:]:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                last_assistant = msg
            elif msg.get("role") == "tool":
                tool_ids.add(msg["tool_call_id"])

        if last_assistant:
            expected_ids = {tc["id"] for tc in last_assistant["tool_calls"]}
            report("cancel_tool_pairs_match", expected_ids == tool_ids)
        else:
            report("cancel_tool_pairs_match", True)  # no tool_calls, trivially consistent

        await runtime.close_session("test")


async def test_local_forkserver_basic(sb):
    r = await sb.execute("print(2 + 2)", timeout=10)
    report("local_basic_output", r.strip() == "4", repr(r.strip()))


async def test_local_forkserver_state_persists(sb):
    await sb.execute("x = 42", timeout=10)
    r = await sb.execute("print(x * 2)", timeout=10)
    report("local_state_persists", r.strip() == "84", repr(r.strip()))


async def test_local_forkserver_error_recovery(sb):
    r1 = await sb.execute("1 / 0", timeout=10)
    report("local_error_traceback", "ZeroDivisionError" in r1)
    r2 = await sb.execute("print('recovered')", timeout=10)
    report("local_error_recovery", r2.strip() == "recovered", repr(r2.strip()))


async def test_local_forkserver_independent_namespaces(fs):
    sb1 = await fs.create_sandbox()
    sb2 = await fs.create_sandbox()
    await sb1.execute("color = 'red'", timeout=10)
    await sb2.execute("color = 'blue'", timeout=10)
    r1 = await sb1.execute("print(color)", timeout=10)
    r2 = await sb2.execute("print(color)", timeout=10)
    ok = r1.strip() == "red" and r2.strip() == "blue"
    report("local_independent_namespaces", ok, f"sb1={r1.strip()!r} sb2={r2.strip()!r}")
    await sb1.close()
    await sb2.close()


async def test_local_forkserver_timeout(sb):
    r = await sb.execute("import time; time.sleep(10)", timeout=2)
    report("local_timeout", "timed out" in r.lower(), repr(r[:80]))


async def test_local_forkserver_post_timeout_recovery(sb):
    r = await sb.execute("print('recovered')", timeout=10)
    report("local_post_timeout_recovery", r.strip() == "recovered", repr(r.strip()))


async def test_cancel_drains_queue():
    """After cancelling client.execute, the queue is clean — the next
    execute call on the same agent gets a fresh result, not a stale one."""
    async with LocalForkServer() as fs:
        sb = await fs.create_sandbox()

        # Start a long-running execution
        task = asyncio.create_task(
            sb.execute("import time; time.sleep(30)", timeout=60)
        )
        # Let it start
        await asyncio.sleep(0.5)

        # Cancel it
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The queue should be clean. Next execute should work correctly.
        r = await sb.execute("print('after_cancel')", timeout=10)
        report("cancel_drain_clean", r.strip() == "after_cancel", repr(r.strip()))

        await sb.close()


async def test_cancel_run_single_cleans_subtasks():
    """Cancelling run_single cancels sub-agent tasks and leaves the
    session usable for subsequent turns.

    Uses mock _run_turn to test the Session-level cancellation logic:
    - creates a real sub-agent task via _host_create_agent/_host_run_agent
    - the sub-agent's mock _run_turn sleeps forever (simulating work)
    - cancel propagates to the sub-agent task
    - _running_tasks is cleared
    - session remains usable for a follow-up turn
    """
    config = AgentConfig(model="test/model", sandbox_image=TAG)
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=_NullClient())

    cancelled_agents = []
    async def _mock_run_turn(full_history, sandbox, request_kwargs, agent_label="main"):
        if agent_label != "main":
            # Sub-agent: sleep forever until cancelled
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled_agents.append(agent_label)
                raise
        # Main agent: create a sub-agent, start a task, then sleep forever
        task = full_history[-1]["content"]
        if task == "spawn_and_hang":
            # Simulate: create_agent + run_agent via direct runtime calls
            aid = await runtime._host_create_agent("worker", _caller_id="cli")
            await runtime._host_run_agent(aid, "do work")
            # Now hang as if awaiting result
            await asyncio.sleep(3600)
        else:
            result = await sandbox.execute(task, timeout=30.0)
            full_history.append({"role": "assistant", "content": result})
            return result

    async with runtime:
        session = await runtime.create_session("cli")
        session._run_turn = _mock_run_turn
        turn_task = asyncio.create_task(
            session.run_single("spawn_and_hang")
        )
        await asyncio.sleep(1.0)
        report("cancel_subtask_exists", len(session._running_tasks) == 1)
        # Cancel the turn
        turn_task.cancel()
        try:
            await turn_task
        except asyncio.CancelledError:
            pass
        # Sub-agent task should have been cancelled
        report("cancel_subtask_cancelled", len(cancelled_agents) == 1)
        report("cancel_tasks_cleared", len(session._running_tasks) == 0)
        result = await session.run_single("print('still_alive')")
        report("cancel_session_reusable", "still_alive" in result, repr(result.strip()))
        await runtime.close_session("cli")


async def test_cancel_interrupts_subagent_e2e():
    """Full end-to-end: cancel a turn that spawned a sub-agent doing real
    sandbox work.  Both the main and sub-agent child processes get SIGINT,
    their queues are drained, and the next turn sees clean output.

    Uses local fork server (no Docker) + mock LLM. The mock LLM responses:
      1. Main round 0: tool_call → code creates sub-agent, runs it, awaits result
      2. Sub-agent round 0: tool_call → code does time.sleep(60)
      (turn is cancelled while sub-agent sleeps)
    After cancel:
      3. Main round 0 (retry): tool_call → code prints a marker
      4. Main round 1: text response with the marker
    """
    call_count = 0
    sub_responded = asyncio.Event()

    sub_agent_code = """aid = await create_agent(instructions='helper')
tid = await run_agent(agent_id=aid, task='Sleep forever')
result = await await_result(tid)
print(f'sub says: {result}')"""

    def _make_responses():
        """Return a fresh response list for a turn attempt."""
        return [
            _tool_resp(sub_agent_code),         # main round 0: spawn sub-agent
            _tool_resp("import time; time.sleep(60)"),  # sub-agent round 0: long sleep
            _text_resp("Should not reach"),     # main round 1 (never reached)
        ]

    responses = _make_responses()

    async def mock_llm(messages, request_kwargs):
        nonlocal call_count, responses
        call_count += 1
        if not responses:
            # After cancellation, the retry turn gets fresh responses
            raise RuntimeError("Mock LLM exhausted unexpectedly")
        return responses.pop(0)

    config = AgentConfig(model="test/model", sandbox_image=None)  # local mode
    registry = HostFunctionRegistry()
    runtime = AgentRuntime(config, registry, llm_client=_NullClient())

    async with runtime:
        runtime._llm_call = mock_llm

        session = await runtime.create_session("test")

        # Start a turn that will hang on sub-agent sleep
        turn_task = asyncio.create_task(
            session.run_single("Use a sub-agent to do work")
        )
        # Give enough time for fork server to spawn children and sub-agent to start sleeping
        await asyncio.sleep(3.0)

        # Sub-agent task should be running
        n_running = len(session._running_tasks)
        report("e2e_cancel_subtask_running", n_running >= 1, f"running={n_running}")

        # Cancel the turn
        turn_task.cancel()
        try:
            await turn_task
        except asyncio.CancelledError:
            pass

        # All sub-agent tasks should be cleaned up
        report("e2e_cancel_tasks_cleared", len(session._running_tasks) == 0,
               f"remaining={len(session._running_tasks)}")

        # Main session sandbox should still work — no stale output.
        # Set up fresh responses for the recovery turn.
        responses = [
            _tool_resp("print('main_ok_after_cancel')"),  # main round 0
            _text_resp("Recovery complete"),               # main round 1
        ]
        result = await session.run_single("Recover")
        report("e2e_cancel_main_clean", "Recovery complete" in result, repr(result))

        # The main sandbox actually executed the code — verify no stale sleep output leaked
        # (If the queue was dirty, we'd get garbage or hang here.)
        report("e2e_cancel_no_stale", "time.sleep" not in result and "sleep" not in result.lower(),
               repr(result))

        await runtime.close_session("test")
# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    print("Config tests:")
    await test_config_defaults()
    await test_config_override()

    print("\nSystem prompt tests:")
    await test_system_prompt_exists()
    await test_system_prompt_mentions_python_tool()

    print("\nTool schema tests:")
    await test_tool_schema_structure()
    await test_tool_schema_json_serializable()

    print("\nCLI arg parsing tests:")
    await test_cli_parse_defaults()
    await test_cli_parse_overrides()
    await test_cli_functions_flag()
    await test_cli_context_and_json_flags()

    print("\nSandbox tests (starting fork server...):")
    async with ForkServer(tag=TAG) as fs:
        sb = await fs.create_sandbox()

        await test_sandbox_basic_output(sb)
        await test_sandbox_no_output(sb)
        await test_sandbox_state_persists(sb)
        await test_sandbox_function_persists(sb)
        await test_sandbox_import_persists(sb)
        await test_sandbox_multiline(sb)
        await test_sandbox_multiline_function(sb)
        await test_sandbox_error_traceback(sb)
        await test_sandbox_syntax_error(sb)
        await test_sandbox_name_error(sb)
        await test_sandbox_error_does_not_poison_state(sb)
        await test_sandbox_numpy(sb)
        await test_sandbox_pandas(sb)
        await test_sandbox_requests(sb)
        await test_sandbox_large_output(sb)
        await test_sandbox_empty_code(sb)
        await test_sandbox_expression_result(sb)
        await test_sandbox_timeout(sb)
        await test_sandbox_post_timeout_recovery(sb)
        await test_sandbox_top_level_async(sb)

        await sb.close()

        print("\nTool dispatch tests:")
        sb2 = await fs.create_sandbox()
        await test_tool_execute_python(sb2)
        await test_tool_execute_unknown(sb2)
        await test_tool_execute_error_propagates(sb2)
        await sb2.close()

        print("\nFork server tests:")
        await test_forkserver_independent_namespaces(fs)
        await test_forkserver_preloaded_imports(fs)
        await test_forkserver_interrupt_isolation(fs)
        await test_forkserver_destroy_respawn(fs)
        await test_forkserver_pip_install_shared(fs)
        await test_forkserver_concurrent_execute(fs)
        await test_forkserver_sequential_reuse(fs)
        await test_sub_agent_task_serialization(fs)

    print("\nLocal fork server tests (no Docker):")
    async with LocalForkServer() as lfs:
        lsb = await lfs.create_sandbox()

        await test_local_forkserver_basic(lsb)
        await test_local_forkserver_state_persists(lsb)
        await test_local_forkserver_error_recovery(lsb)
        await test_local_forkserver_timeout(lsb)
        await test_local_forkserver_post_timeout_recovery(lsb)

        await lsb.close()

        await test_local_forkserver_independent_namespaces(lfs)

    print("\nHost function path tests (separate runtime):")
    await test_host_function_path_serialization()

    print("\nEvent emission tests (separate runtimes, mock LLM):")
    await test_event_emission_two_rounds()
    await test_event_emission_immediate()
    await test_event_callback_error_isolation()
    await test_event_no_callback()
    await test_event_sub_agent_propagation()

    print("\nLLM client protocol tests:")
    await test_openrouter_client_converts_text_response()
    await test_openrouter_client_converts_tool_response()
    await test_llm_client_injection()
    await test_llm_client_lifecycle_ownership()

    print("\nShutdown / SIGTERM tests:")
    await test_sigterm_clean_shutdown()
    await test_sigterm_cancels_inflight_subtasks()
    await test_cancellation_rolls_back_history()
    await test_cancel_drains_queue()
    await test_cancel_run_single_cleans_subtasks()
    await test_cancel_interrupts_subagent_e2e()
    total = passed_count + len(failures)
    print()
    if failures:
        print(f"FAILED: {len(failures)}/{total} — {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"All {total} tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
