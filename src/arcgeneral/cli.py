import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from arcgeneral.agent import run_agent, run_session
from arcgeneral.config import AgentConfig
from arcgeneral.host_functions import HostFunctionRegistry
from arcgeneral.internet_extract import execute_internet_extract
from arcgeneral.internet_search import execute_internet_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM agent with stateful IPython REPL")
    parser.add_argument("message", type=str, nargs="?", default=None, help="User message (omit for interactive session)")
    parser.add_argument("--model", type=str, default="z-ai/glm-5", help="Model name")
    parser.add_argument("--api-key-env-var", type=str, default="OPENROUTER_API_KEY", help="Env var name for API key")
    parser.add_argument("--image", type=str, default="arcgeneral:sandbox", help="Docker image tag for sandbox")
    parser.add_argument("--timeout", type=float, default=120.0, help="Code execution timeout in seconds")
    parser.add_argument("--max-rounds", type=int, default=50, help="Max tool loop iterations")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("arcgeneral").setLevel(logging.INFO)
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
`after_date` (YYYY-MM-DD) is a soft signal — older results may still appear.
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
        host_functions=registry,
    )

    if args.message is not None:
        result = asyncio.run(run_agent(config, args.message))
        print(result)
    else:
        asyncio.run(run_session(config))


if __name__ == "__main__":
    main()
