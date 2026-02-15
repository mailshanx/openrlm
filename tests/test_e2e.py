"""End-to-end integration tests for arcgeneral.

Tests everything except the agent loop (run_agent), which requires API keys.

Requires Docker running and the arcgeneral:sandbox image built:
    uv run python -m ipybox build -t arcgeneral:sandbox -d sandbox-deps.txt --root

Run:
    uv run python tests/test_e2e.py
"""

import asyncio
import json
import sys

from arcgeneral.agent import DEFAULT_SYSTEM_PROMPT
from arcgeneral.config import AgentConfig
from arcgeneral.sandbox import Sandbox
from arcgeneral.tool import PYTHON_TOOL_SCHEMA, execute_tool

TAG = "arcgeneral:sandbox"

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
        c.api_key_env_var == "OPENROUTER_API_KEY",
        c.system_prompt is None,
        c.sandbox_image == "arcgeneral:sandbox",
        c.code_timeout == 120.0,
        c.max_tool_rounds == 50,
        c.temperature is None,
    ]))


async def test_config_override():
    c = AgentConfig(
        model="openai/gpt-4o-mini",
        api_key_env_var="MY_KEY",
        system_prompt="You are a cat.",
        sandbox_image="custom:latest",
        code_timeout=30.0,
        max_tool_rounds=5,
        temperature=0.7,
    )
    report("config_override", all([
        c.model == "openai/gpt-4o-mini",
        c.api_key_env_var == "MY_KEY",
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
        report("cli_default_model", args.model == "qwen/qwen3-coder-next")
        report("cli_default_api_key_env", args.api_key_env_var == "OPENROUTER_API_KEY")
        report("cli_default_image", args.image == "arcgeneral:sandbox")
        report("cli_default_timeout", args.timeout == 120.0)
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
            "--api-key-env-var", "MY_KEY",
            "--image", "custom:v2",
            "--timeout", "30.5",
            "--max-rounds", "10",
            "--env-file", "prod.env",
            "--verbose",
        ]
        args = parse_args()
        report("cli_override_message", args.message == "compute stuff")
        report("cli_override_model", args.model == "openai/gpt-4o-mini")
        report("cli_override_api_key_env", args.api_key_env_var == "MY_KEY")
        report("cli_override_image", args.image == "custom:v2")
        report("cli_override_timeout", args.timeout == 30.5)
        report("cli_override_max_rounds", args.max_rounds == 10)
        report("cli_override_env_file", args.env_file == "prod.env")
        report("cli_override_verbose", args.verbose is True)
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


async def test_sandbox_matplotlib_import(sb: Sandbox):
    result = await sb.execute("import matplotlib; print(matplotlib.__version__)")
    report("sandbox_matplotlib", result.strip() != "" and "Error" not in result, repr(result[:60]))


async def test_sandbox_scipy(sb: Sandbox):
    result = await sb.execute("from scipy import constants; print(int(constants.c))")
    report("sandbox_scipy", "299792458" in result, repr(result))


async def test_sandbox_scikit_learn(sb: Sandbox):
    result = await sb.execute("import sklearn; print(sklearn.__version__)")
    report("sandbox_sklearn", result.strip() != "" and "Error" not in result, repr(result[:60]))


async def test_sandbox_seaborn(sb: Sandbox):
    result = await sb.execute("import seaborn; print(seaborn.__version__)")
    report("sandbox_seaborn", result.strip() != "" and "Error" not in result, repr(result[:60]))


async def test_sandbox_pillow(sb: Sandbox):
    result = await sb.execute("from PIL import Image; print(Image.__name__)")
    report("sandbox_pillow", "Image" in result, repr(result[:60]))


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
    """IPython prints the result of a bare expression."""
    result = await sb.execute("2 ** 10")
    report("sandbox_expression_result", "1024" in result, repr(result))


async def test_sandbox_timeout(sb: Sandbox):
    result = await sb.execute("import time; time.sleep(10)", timeout=2)
    report("sandbox_timeout", "timed out" in result.lower() or "Timeout" in result, repr(result[:80]))


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

    print("\nSandbox tests (starting container...):")
    async with Sandbox(tag=TAG) as sb:
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
        await test_sandbox_matplotlib_import(sb)
        await test_sandbox_scipy(sb)
        await test_sandbox_scikit_learn(sb)
        await test_sandbox_seaborn(sb)
        await test_sandbox_pillow(sb)
        await test_sandbox_requests(sb)
        await test_sandbox_large_output(sb)
        await test_sandbox_empty_code(sb)
        await test_sandbox_expression_result(sb)
        await test_sandbox_timeout(sb)

        print("\nTool dispatch tests:")
        await test_tool_execute_python(sb)
        await test_tool_execute_unknown(sb)
        await test_tool_execute_error_propagates(sb)

    total = passed_count + len(failures)
    print()
    if failures:
        print(f"FAILED: {len(failures)}/{total} — {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"All {total} tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
