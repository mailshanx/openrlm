import argparse
import asyncio
import logging
import signal
from pathlib import Path
from dotenv import load_dotenv
from arcgeneral.config import AgentConfig
from arcgeneral.host_functions import HostFunctionRegistry
from arcgeneral.contrib import internet_extract, internet_search
from arcgeneral.sandbox import cleanup_orphaned_containers
from arcgeneral.agent import AgentRuntime

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM agent with stateful IPython REPL")
    parser.add_argument("message", type=str, nargs="?", default=None, help="User message (omit for interactive session)")
    parser.add_argument("--model", type=str, default="z-ai/glm-5", help="Model name")
    parser.add_argument("--api-key-env-var", type=str, default="OPENROUTER_API_KEY", help="Env var name for API key")
    parser.add_argument("--image", type=str, default=None, help="Docker image tag for sandbox (omit for local mode)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Code execution timeout in seconds")
    parser.add_argument("--max-rounds", type=int, default=50, help="Max tool loop iterations")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--workspace", type=str, default=None, help="Working directory shared with agents (default: cwd)")
    parser.add_argument("--log-file", type=str, default=str(Path.home() / "Downloads" / "arcgeneral.log"), help="Log file path")
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
    internet_extract.register(registry)
    internet_search.register(registry)
    config = AgentConfig(
        model=args.model,
        api_key_env_var=args.api_key_env_var,
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

                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(sig, shutdown.set)

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
                    result = await session.run_single(stripped)
                    print(result)
                    print()
                await runtime.close_session("cli")
        asyncio.run(_session())


if __name__ == "__main__":
    main()
