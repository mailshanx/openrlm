"""internet_search — search the web and return relevant excerpts with source URLs.

Uses the Parallel API (parallel-web) to search the web. Returns results with
URLs, titles, publish dates, and relevant excerpts as markdown.

At least one of `objective` or `search_queries` is required; both recommended.

Requires PARALLEL_API_KEY in the environment.
"""

import json
import logging
import os

from parallel import Parallel


logger = logging.getLogger(__name__)


async def execute_internet_search(
    objective: str = "",
    search_queries: str = "",
    include_domains: str = "",
    exclude_domains: str = "",
    after_date: str = "",
) -> str:
    """Search the web and return relevant excerpts with source URLs.

    Calls the Parallel API to search the web. Results include URLs, titles,
    publish dates, and relevant excerpts formatted as markdown.

    Args:
        objective: What you need and why. Include task context and preferred sources.
        search_queries: Semicolon-separated queries with optional boolean operators.
        include_domains: Semicolon-separated domains to restrict results to.
        exclude_domains: Semicolon-separated domains to exclude from results.
        after_date: Prefer content after this date (YYYY-MM-DD). Soft signal.

    Returns:
        JSON string with key "results", a list of
        {url, title, publish_date, excerpts}.

    Raises:
        RuntimeError: If the API key is missing, no search parameters are given,
            or the Parallel API returns an error.
    """
    api_key = os.environ.get("PARALLEL_API_KEY")
    if not api_key:
        raise RuntimeError("PARALLEL_API_KEY is not set")

    query_list = [q.strip() for q in search_queries.split(";") if q.strip()] if search_queries else []

    if not objective and not query_list:
        raise RuntimeError("At least one of 'objective' or 'search_queries' is required.")

    logger.info(
        "internet_search: objective='%s', queries=%s",
        objective[:100] if objective else "", query_list,
    )

    client = Parallel(api_key=api_key)

    include_list = [d.strip() for d in include_domains.split(";") if d.strip()] if include_domains else []
    exclude_list = [d.strip() for d in exclude_domains.split(";") if d.strip()] if exclude_domains else []

    source_policy = None
    if include_list or exclude_list or after_date:
        source_policy = {}
        if include_list:
            source_policy["include_domains"] = include_list
        if exclude_list:
            source_policy["exclude_domains"] = exclude_list
        if after_date:
            source_policy["after_date"] = after_date

    api_kwargs = {
        "mode": "agentic",
        "max_results": 20,
        "excerpts": {"max_chars_per_result": 5000, "max_chars_total": 25000},
    }
    if objective:
        api_kwargs["objective"] = objective
    if query_list:
        api_kwargs["search_queries"] = query_list
    if source_policy:
        api_kwargs["source_policy"] = source_policy

    search_response = client.beta.search(**api_kwargs)

    results = []
    for result in search_response.results:
        results.append({
            "url": result.url,
            "title": result.title,
            "publish_date": result.publish_date,
            "excerpts": result.excerpts or [],
        })

    if not results:
        raise RuntimeError("No results found. Try broadening your queries or adjusting filters.")

    logger.info("internet_search returned %d results", len(results))

    return json.dumps({"results": results}, ensure_ascii=False)
