"""internet_extract tool — fetch a web page or PDF and return its content as markdown.

Uses the Parallel API (parallel-web) to fetch, render, strip boilerplate,
and convert HTML to markdown.  Two modes:

- **Focused extraction** (default): provide `objective` and/or `search_queries`
  to receive only the relevant excerpts.
- **Full extraction**: set `full_content=True` to get the entire page as markdown.

At least one of `objective`, `search_queries`, or `full_content=True` is required.

Requires PARALLEL_API_KEY in the environment.
"""

import json
import logging
import os

from parallel import Parallel


logger = logging.getLogger(__name__)




async def execute_internet_extract(
    url: str,
    objective: str = "",
    search_queries: str = "",
    full_content: bool = False,
) -> str:
    """Execute an internet_extract tool call. Returns a JSON string with the results.

    Calls the Parallel API to fetch and extract content from the given URL.
    In focused mode, returns relevant excerpts. In full mode, returns the
    entire page content as markdown.

    Args:
        url: Public URL to fetch.
        objective: What information is needed. Guides excerpt selection.
        search_queries: Semicolon-separated search terms.
        full_content: If True, return full page content instead of excerpts.

    Returns:
        JSON string with keys: url, title, publish_date, and either
        excerpts (focused mode) or full_content (full mode).

    Raises:
        RuntimeError: If the API key is missing, no extraction mode is specified,
            or the Parallel API returns an error.
    """
    api_key = os.environ.get("PARALLEL_API_KEY")
    if not api_key:
        raise RuntimeError("PARALLEL_API_KEY is not set")

    query_list = [q.strip() for q in search_queries.split(";") if q.strip()] if search_queries else []

    if not objective and not query_list and not full_content:
        raise RuntimeError("At least one of 'objective', 'search_queries', or 'full_content=True' is required.")

    logger.info(
        "internet_extract: url='%s', objective='%s', queries=%s, full_content=%s",
        url, objective[:100] if objective else "", query_list, full_content,
    )

    client = Parallel(api_key=api_key)

    api_kwargs = {
        "urls": [url],
        "excerpts": not full_content,
        "full_content": full_content,
        "betas": ["search-extract-2025-10-10"],
    }
    if objective:
        api_kwargs["objective"] = objective
    if query_list:
        api_kwargs["search_queries"] = query_list

    extract_response = client.beta.extract(**api_kwargs)

    if extract_response.errors:
        error_msgs = [f"{e.url}: {e.message}" for e in extract_response.errors]
        raise RuntimeError(f"Extraction failed: {'; '.join(error_msgs)}")

    if not extract_response.results:
        raise RuntimeError(f"No content could be extracted from {url}")

    result = extract_response.results[0]

    if full_content and result.full_content:
        content = result.full_content
        content_key = "full_content"
    elif result.excerpts:
        content = "\n\n".join(result.excerpts) if isinstance(result.excerpts, list) else result.excerpts
        content_key = "excerpts"
    else:
        content = ""
        content_key = "excerpts"

    output = {
        "url": result.url,
        "title": result.title,
        "publish_date": result.publish_date,
        content_key: content,
    }

    logger.info("internet_extract completed for %s", url)

    return json.dumps(output, ensure_ascii=False)
