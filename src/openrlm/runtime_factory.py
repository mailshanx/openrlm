"""Convenience factories for building a fully-wired AgentRuntime.

Use this when you want the default built-in harness (LLM client selection,
host function loading, API key resolution) without coupling to the CLI's
argparse or UI concerns.
"""

import importlib
import importlib.util
from pathlib import Path

from openrlm.agent import AgentRuntime
from openrlm.config import AgentConfig
from openrlm.host_functions import HostFunctionRegistry
from openrlm.llm import (
    LLMClient,
    OpenRouterClient,
    AnthropicClient,
    default_api_key_resolver,
)


def _load_module_from_file(filepath: Path) -> object:
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Error: could not load {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _register_module(registry: HostFunctionRegistry, module: object, source: str) -> None:
    """Call register(registry) on a loaded module, or raise SystemExit."""
    register_fn = getattr(module, "register", None)
    if register_fn is None:
        raise SystemExit(
            f"Error: {source} has no register() function. "
            f"Expected: def register(registry: HostFunctionRegistry) -> None"
        )
    register_fn(registry)


def load_functions(registry: HostFunctionRegistry, specs: list[str]) -> None:
    """Load custom functions from file paths, directories, or dotted module names.

    Each spec is one of:
      - A directory path  → load all .py files in it that have register()
      - A .py file path   → load that file
      - A dotted name     → importlib.import_module()
    """
    for spec in specs:
        p = Path(spec).expanduser()
        if p.is_dir():
            py_files = sorted(p.glob("*.py"))
            if not py_files:
                raise SystemExit(f"Error: no .py files found in {p}")
            for f in py_files:
                if f.name.startswith("_"):
                    continue
                module = _load_module_from_file(f)
                if hasattr(module, "register"):
                    _register_module(registry, module, str(f))
        elif p.is_file() or spec.endswith(".py"):
            if not p.is_file():
                raise SystemExit(f"Error: file not found: {spec}")
            module = _load_module_from_file(p)
            _register_module(registry, module, spec)
        else:
            # Dotted module name — standard import
            try:
                module = importlib.import_module(spec)
            except ImportError as e:
                raise SystemExit(
                    f"Error: could not import function module {spec!r}: {e}\n"
                    f"Hint: if installed via 'uv tool', re-install with: "
                    f"uv tool install openrlm --with <package>"
                ) from e
            _register_module(registry, module, spec)


def build_llm_client(
    provider: str,
    *,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "auto",
    text_verbosity: str = "medium",
) -> LLMClient:
    """Build an LLM client for the given provider name.

    The Codex-specific keyword arguments are only used when provider is
    "openai-codex" and are ignored otherwise.
    """
    if provider == "anthropic":
        return AnthropicClient()
    elif provider == "openai-codex":
        from openrlm.codex import CodexClient
        return CodexClient(
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            text_verbosity=text_verbosity,
        )
    else:
        return OpenRouterClient()


def build_runtime(
    *,
    model: str = "openai/gpt-5.2",
    provider: str = "openrouter",
    image: str | None = None,
    timeout: float = 3600.0,
    max_rounds: int = 50,
    workspace: str | Path | None = None,
    functions: list[str] | None = None,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "auto",
    text_verbosity: str = "medium",
) -> AgentRuntime:
    """Build a fully-wired AgentRuntime from simple parameters.

    This is the main entry point for programmatic use. It handles host
    function loading, API key resolution, LLM client selection, and
    AgentConfig construction — the same wiring the CLI does internally.

    The returned runtime must be used as an async context manager::

        runtime = build_runtime(provider="anthropic", model="claude-sonnet-4-5")
        async with runtime:
            session = await runtime.create_session("my-session", on_event=handler)
            result = await session.run_single("hello")
            await runtime.close_session("my-session")
    """
    registry = HostFunctionRegistry()
    if functions:
        load_functions(registry, functions)

    resolver = default_api_key_resolver()
    ws = Path(workspace).resolve() if workspace else Path.cwd()
    config = AgentConfig(
        model=model,
        get_api_key=lambda: resolver(provider),
        sandbox_image=image,
        code_timeout=timeout,
        max_tool_rounds=max_rounds,
        sandbox_binds={str(ws): "workspace"},
    )

    llm_client = build_llm_client(
        provider,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        text_verbosity=text_verbosity,
    )
    return AgentRuntime(config, registry, llm_client=llm_client)
