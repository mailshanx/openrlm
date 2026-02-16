import argparse
import asyncio
import logging
import signal
from pathlib import Path
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from arcgeneral.config import AgentConfig
from arcgeneral.host_functions import HostFunctionRegistry
from arcgeneral.internet_extract import execute_internet_extract
from arcgeneral.internet_search import execute_internet_search
from arcgeneral.sandbox import cleanup_orphaned_containers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM agent with stateful IPython REPL")
    parser.add_argument("message", type=str, nargs="?", default=None, help="User message (omit for interactive session)")
    parser.add_argument("--model", type=str, default="z-ai/glm-5", help="Model name")
    parser.add_argument("--api-key-env-var", type=str, default="OPENROUTER_API_KEY", help="Env var name for API key")
    parser.add_argument("--image", type=str, default="arcgeneral:sandbox", help="Docker image tag for sandbox")
    parser.add_argument("--timeout", type=float, default=420.0, help="Code execution timeout in seconds")
    parser.add_argument("--max-rounds", type=int, default=50, help="Max tool loop iterations")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    load_dotenv(args.env_file)

    registry = HostFunctionRegistry()
    registry.register(
        "internet_extract",
        execute_internet_extract,
        description="""\
Fetch a web page or PDF and return its content as markdown.
Two modes: focused (provide `objective` and/or `search_queries` for relevant excerpts) \
or full (`full_content=True` for the entire page).
At least one of `objective`, `search_queries`, or `full_content=True` is required.
Returns a JSON string with keys: url, title, publish_date, and either excerpts or full_content.

Example: page = await internet_extract(url='https://example.com/report.pdf', objective='key findings')""",
    )
    registry.register(
        "internet_search",
        execute_internet_search,
        description="""\
Search the web and return relevant excerpts with source URLs.
At least one of `objective` or `search_queries` required; both recommended.
Query syntax (semicolon-separated): AND, OR, "exact phrase", -exclude, wildcard*
Use `include_domains`/`exclude_domains` instead of site: operators in queries.
`after_date` (YYYY-MM-DD) is a soft signal \u2014 older results may still appear.
Returns a JSON string with key 'results', a list of {url, title, publish_date, excerpts}.

Example: results = await internet_search(objective='recent advances in fusion energy', search_queries='fusion energy 2025; tokamak breakthrough')""",
    )
    config = AgentConfig(
        model=args.model,
        api_key_env_var=args.api_key_env_var,
        sandbox_image=args.image,
        code_timeout=args.timeout,
        max_tool_rounds=args.max_rounds,
        sandbox_binds={str(Path.home() / "Downloads"): "downloads"},
    )

    # create_agent/run_agent are registered by AgentRuntime.__init__
    runtime = AgentRuntime(config, registry)

    if args.message is not None:
        async def _run():
            await cleanup_orphaned_containers()
            async with runtime:
                return await runtime.run_single(args.message)
        print(asyncio.run(_run()))
    else:
        async def _session():
            await cleanup_orphaned_containers()
            async with runtime:
                loop = asyncio.get_event_loop()
                loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.ensure_future(runtime.__aexit__(None, None, None)))
                await runtime.run_session()
        asyncio.run(_session())


if __name__ == "__main__":
    main()
