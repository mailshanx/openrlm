import argparse
import asyncio
import importlib
import importlib.util
import logging
import signal
import textwrap
import time
from pathlib import Path
from dotenv import load_dotenv
from arcgeneral.config import AgentConfig
from arcgeneral.host_functions import HostFunctionRegistry
from arcgeneral.sandbox import cleanup_orphaned_containers
from arcgeneral.agent import AgentRuntime


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


def _load_functions(registry: HostFunctionRegistry, specs: list[str]) -> None:
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
                    f"uv tool install arcgeneral --with <package>"
                ) from e
            _register_module(registry, module, spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM agent with stateful IPython REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              arcgeneral "summarize this repo"
              arcgeneral --provider anthropic --model claude-sonnet-4-5 "explain main.py"
              arcgeneral --functions ./my-tools "use my_search to find X"
              arcgeneral --functions ./contrib "search for X"
              arcgeneral  # interactive session

            environment:
              API keys are read from provider-specific environment variables:
                openrouter   → OPENROUTER_API_KEY (default)
                anthropic    → ANTHROPIC_API_KEY
                openai       → OPENAI_API_KEY
                google       → GEMINI_API_KEY
              A .env file in the current directory is loaded automatically.
        """),
    )
    parser.add_argument("message", type=str, nargs="?", default=None,
                        help="User message (omit for interactive session)")
    parser.add_argument("--model", type=str, default="z-ai/glm-5",
                        help="Model identifier (default: %(default)s)")
    parser.add_argument("--provider", type=str, default="openrouter",
                        help="LLM provider — determines API key and endpoint (default: %(default)s)")
    parser.add_argument("--image", type=str, default=None,
                        help="Docker image tag for sandbox (omit for local mode)")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="Code execution timeout in seconds (default: %(default)s)")
    parser.add_argument("--max-rounds", type=int, default=50,
                        help="Max tool loop iterations (default: %(default)s)")
    parser.add_argument("--env-file", type=str, default=".env",
                        help="Path to .env file (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Working directory shared with agents (default: cwd)")
    parser.add_argument("--log-file", type=str,
                        default=str(Path.home() / "Downloads" / "arcgeneral.log"),
                        help="Log file path (default: %(default)s)")
    parser.add_argument("--functions", type=str, default=None,
                        metavar="PATH",
                        help="Directory, .py file, or dotted module name to load custom functions from "
                             "(comma-separated for multiple)")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    arc_logger = logging.getLogger("arcgeneral")
    arc_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    arc_logger.addHandler(file_handler)
    arc_logger.propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    load_dotenv(args.env_file)

    registry = HostFunctionRegistry()
    if args.functions:
        modules = [m.strip() for m in args.functions.split(",") if m.strip()]
        _load_functions(registry, modules)

    config = AgentConfig(
        model=args.model,
        provider=args.provider,
        sandbox_image=args.image,  # None = local mode
        code_timeout=args.timeout,
        max_tool_rounds=args.max_rounds,
        sandbox_binds={str(Path(args.workspace).resolve() if args.workspace else Path.cwd()): "workspace"},
    )

    # create_agent/run_agent are registered by AgentRuntime.__init__
    runtime = AgentRuntime(config, registry)

    if args.message is not None:
        async def _run():
            if args.image:
                await cleanup_orphaned_containers()
            async with runtime:
                session = await runtime.create_session("cli")
                result = await session.run_single(args.message)
                await runtime.close_session("cli")
                return result
        print(asyncio.run(_run()))
    else:
        async def _session():
            if args.image:
                await cleanup_orphaned_containers()
            async with runtime:
                session = await runtime.create_session("cli")
                loop = asyncio.get_event_loop()
                shutdown = asyncio.Event()
                turn_task: asyncio.Task | None = None
                last_interrupt = 0.0

                def _on_signal():
                    nonlocal last_interrupt
                    now = time.monotonic()
                    if turn_task is not None and not turn_task.done():
                        # Turn is running — cancel it
                        turn_task.cancel()
                        if now - last_interrupt < 1.0:
                            # Double Ctrl-C while turn running — exit
                            shutdown.set()
                        last_interrupt = now
                    else:
                        # No turn running — exit
                        shutdown.set()
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(sig, _on_signal)
                mode = "docker" if args.image else "local"
                print(f"arcgeneral session started ({mode} mode). Type 'quit' or 'exit' to end.\n")
                while not shutdown.is_set():
                    try:
                        user_input = await loop.run_in_executor(None, input, ">>> ")
                    except EOFError:
                        break
                    stripped = user_input.strip()
                    if stripped.lower() in ("quit", "exit"):
                        break
                    if not stripped:
                        continue
                    turn_task = asyncio.create_task(session.run_single(stripped))
                    try:
                        result = await turn_task
                        print(result)
                        print()
                    except asyncio.CancelledError:
                        print("\n(interrupted)")
                    finally:
                        turn_task = None
                await runtime.close_session("cli")
        asyncio.run(_session())


if __name__ == "__main__":
    main()
