"""MediaWiki API wrapper exposing a single search_wikipedia(query) function."""

import urllib.parse

import httpx

WIKI_API = "https://en.wikipedia.org/w/api.php"
EXTRACT_LIMIT = 2500
HEADERS = {"User-Agent": "wiki-qa/1.0 (https://github.com/blackmamba/wiki-qa) httpx/0.27"}


def search_wikipedia(query: str) -> dict:
    """Search Wikipedia; return the best-match article's title, URL, and text extract."""
    search_resp = httpx.get(
        WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
            "utf8": 1,
        },
        headers=HEADERS,
        timeout=10,
    )
    search_resp.raise_for_status()
    results = search_resp.json().get("query", {}).get("search", [])

    if not results:
        return {
            "title": None,
            "url": None,
            "extract": "No Wikipedia article found for this query. Try a different search term.",
        }

    title = results[0]["title"]

    extract_resp = httpx.get(
        WIKI_API,
        params={
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json",
            "utf8": 1,
        },
        headers=HEADERS,
        timeout=10,
    )
    extract_resp.raise_for_status()
    pages = extract_resp.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    extract = page.get("extract") or ""

    if len(extract) > EXTRACT_LIMIT:
        extract = extract[:EXTRACT_LIMIT] + " ...[truncated]"

    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
    return {"title": title, "url": url, "extract": extract or "No content available."}
