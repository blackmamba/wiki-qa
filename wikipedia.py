"""MediaWiki API wrapper exposing a single ``search_wikipedia(query)`` function.

Uses a two-request pattern:
  1. Search for candidate article titles via the ``list=search`` API.
  2. Fetch the intro extract of the top-ranked article via ``prop=extracts``.

Returns a plain dict with keys ``title``, ``url``, and ``extract``.
Network and parse errors are caught and returned as an error dict so callers
(the agentic loop) never have to handle exceptions from this module.
"""

import urllib.parse
from typing import Optional, TypedDict

import httpx

WIKI_API = "https://en.wikipedia.org/w/api.php"
EXTRACT_LIMIT = 2500
SEARCH_RESULT_LIMIT = 3
HTTP_TIMEOUT = 10  # seconds

# Include the actual httpx version so the User-Agent stays accurate on upgrades.
_UA = f"wiki-qa/1.0 (https://github.com/blackmamba/wiki-qa) httpx/{httpx.__version__}"
HEADERS = {"User-Agent": _UA}


class WikiResult(TypedDict):
    """Shape of the dict returned by :func:`search_wikipedia`."""

    title: Optional[str]
    url: Optional[str]
    extract: str


_ERROR_RESULT: WikiResult = {
    "title": None,
    "url": None,
    "extract": "Wikipedia search is temporarily unavailable. Please try again.",
}

_NO_RESULT: WikiResult = {
    "title": None,
    "url": None,
    "extract": "No Wikipedia article found for this query. Try a different search term.",
}


def _wiki_get(params: dict) -> dict:
    """Issue a GET request to the MediaWiki API and return the parsed JSON body.

    Raises:
        httpx.HTTPError: on any non-2xx response or network failure.
        ValueError: if the response body is not valid JSON.
    """
    resp = httpx.get(WIKI_API, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def search_wikipedia(query: str) -> WikiResult:
    """Search Wikipedia and return the best-matching article.

    Issues up to two requests: one search query to find the top article title,
    then one extract fetch to retrieve its introductory text.

    Args:
        query: A natural-language search query string.

    Returns:
        A :class:`WikiResult` dict with keys:
          - ``title``: Article title, or ``None`` if nothing was found.
          - ``url``:   Full Wikipedia URL, or ``None`` if nothing was found.
          - ``extract``: Up to ``EXTRACT_LIMIT`` characters of the article
            intro, or an explanatory message on error/no-result.
    """
    # ── Step 1: find candidate titles ────────────────────────────────────────
    try:
        search_data = _wiki_get({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": SEARCH_RESULT_LIMIT,
            "format": "json",
            "utf8": 1,
        })
        results = search_data.get("query", {}).get("search", [])
    except (httpx.HTTPError, ValueError):
        return _ERROR_RESULT

    if not results:
        return _NO_RESULT

    title = results[0]["title"]

    # ── Step 2: fetch the article extract ────────────────────────────────────
    try:
        extract_data = _wiki_get({
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json",
            "utf8": 1,
        })
        pages = extract_data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        extract = page.get("extract") or ""
    except (httpx.HTTPError, ValueError, StopIteration):
        return _ERROR_RESULT

    if len(extract) > EXTRACT_LIMIT:
        extract = extract[:EXTRACT_LIMIT] + " ...[truncated]"

    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
    return {"title": title, "url": url, "extract": extract or "No content available."}
