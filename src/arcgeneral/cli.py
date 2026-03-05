import argparse
import asyncio
import json
import dataclasses
import sys
import logging
import signal
import textwrap
import time
from pathlib import Path
from dotenv import load_dotenv
from arcgeneral.sandbox import cleanup_orphaned_containers
from arcgeneral.runtime_factory import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursive Language Model (RLM) agent with persistent IPython REPL.\n"
            "\n"
            "Each agent gets its own stateful IPython environment and can recursively\n"
            "spawn sub-agents, each with their own isolated REPL. Sub-agents are forked\n"
            "processes within a single container (or local process tree), making spawning\n"
            "fast and cheap -- agents can programmatically create large numbers of them.\n"
            "\n"
            "Agents interact with context through symbolic manipulation in the REPL:\n"
            "reading files, transforming data, running computations, and building on\n"
            "previous results within the same session. Custom host functions can be\n"
            "injected for capabilities beyond code execution (web search, APIs, etc.).\n"
            "\n"
            "By default, agents run as local processes. Use --image to run inside a\n"
            "Docker container for isolation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              arcgeneral "summarize this repo"
              arcgeneral --provider anthropic --model claude-sonnet-4-5 "explain main.py"
              arcgeneral --functions ./my-tools "use my_search to find X"
              arcgeneral --functions ./contrib "search for X"
              arcgeneral --context history.json "continue the analysis"
              arcgeneral --json "compute pi" | jq .result
              arcgeneral --image arcgeneral:sandbox "analyze data"  # Docker mode
              arcgeneral  # interactive session
            environment:
              API keys are read from provider-specific environment variables:
                openrouter   -> OPENROUTER_API_KEY (default)
                anthropic    -> ANTHROPIC_API_KEY
                openai       -> OPENAI_API_KEY
                google       -> GEMINI_API_KEY
              A .env file in the current directory is loaded automatically.
        """),
    )
    parser.add_argument("message", type=str, nargs="?", default=None,
                        help="User message (omit for interactive session)")
    parser.add_argument("--model", type=str, default="openai/gpt-5.2",
                        help="Model identifier (default: %(default)s)")
    parser.add_argument("--provider", type=str, default="openrouter",
                        help="LLM provider: 'openrouter' (default), 'anthropic' (direct Anthropic API), or 'openai-codex' (Codex via ChatGPT). Other providers route through OpenRouter.")
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
    parser.add_argument("--context", type=str, default=None,
                        metavar="FILE",
                        help="JSON file with conversation history to prepend "
                             '(array of {"role": "user"|"assistant", "content": "..."})')
    parser.add_argument("--json", action="store_true",
                        help="Output result as JSON on stdout; stream agent events as JSONL on stderr")
    parser.add_argument("--build-image", type=str, nargs="?", const="arcgeneral:sandbox",
                        default=None, metavar="TAG",
                        help="Build the Docker sandbox image and exit (default tag: arcgeneral:sandbox)")
    parser.add_argument("--sandbox-deps", type=str, default=None, metavar="FILE",
                        help="Dependencies file for --build-image (default: sandbox-deps.txt if it exists)")
    # Codex-specific options (only used when --provider openai-codex)
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort for Codex models (default: %(default)s)")
    parser.add_argument("--text-verbosity", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Text verbosity for Codex models (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.build_image:
        from arcgeneral.ipybox.build import build
        deps_file = None
        if args.sandbox_deps:
            deps_file = Path(args.sandbox_deps)
            if not deps_file.is_file():
                raise SystemExit(f"Error: dependencies file not found: {args.sandbox_deps}")
        else:
            default_deps = Path("sandbox-deps.txt")
            if default_deps.is_file():
                deps_file = default_deps
        build(args.build_image, deps_file)
        return

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

    # Load conversation context if provided
    context_messages: list[dict] | None = None
    if args.context:
        context_path = Path(args.context).expanduser()
        if not context_path.is_file():
            raise SystemExit(f"Error: context file not found: {args.context}")
        try:
            raw = json.loads(context_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise SystemExit(f"Error: could not read context file: {e}") from e
        if not isinstance(raw, list):
            raise SystemExit("Error: context file must contain a JSON array of messages")
        valid_roles = {"user", "assistant"}
        for i, msg in enumerate(raw):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise SystemExit(f"Error: context message {i} must have 'role' and 'content' fields")
            if msg["role"] not in valid_roles:
                raise SystemExit(f"Error: context message {i} has invalid role {msg['role']!r} (expected user or assistant)")
        context_messages = raw

    runtime = build_runtime(
        model=args.model,
        provider=args.provider,
        image=args.image,
        timeout=args.timeout,
        max_rounds=args.max_rounds,
        workspace=args.workspace,
        functions=[m.strip() for m in args.functions.split(",") if m.strip()] if args.functions else None,
        reasoning_effort=args.reasoning_effort,
        text_verbosity=args.text_verbosity,
    )

    if args.message is None and args.json:
        raise SystemExit("Error: --json requires a message argument (not interactive mode)")
    if args.message is not None:
        async def _run():
            if args.image:
                await cleanup_orphaned_containers()
            on_event = None
            if args.json:
                def on_event(event):
                    line = json.dumps({"type": type(event).__name__, **dataclasses.asdict(event)})
                    sys.stderr.write(line + "\n")
                    sys.stderr.flush()
            async with runtime:
                session = await runtime.create_session("cli", on_event=on_event, context_messages=context_messages)
                result = await session.run_single(args.message)
                await runtime.close_session("cli")
                return result
        if args.json:
            try:
                result = asyncio.run(_run())
                print(json.dumps({"result": result, "error": None}))
            except Exception as e:
                print(json.dumps({"result": None, "error": str(e)}))
        else:
            print(asyncio.run(_run()))
    else:
        async def _session():
            if args.image:
                await cleanup_orphaned_containers()
            async with runtime:
                session = await runtime.create_session("cli", context_messages=context_messages)
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
