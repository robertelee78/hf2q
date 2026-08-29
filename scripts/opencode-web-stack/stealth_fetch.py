"""Isolated Scrapling worker for one bounded Cloudflare-aware fetch.

The parent service owns the wall-clock timeout and terminates this process
group if Scrapling's recursive challenge solver does not converge.
"""

import json
import sys

import trafilatura
from crawl4ai.antibot_detector import is_blocked
from lxml import html as lxml_html
from scrapling.fetchers import StealthyFetcher

from search_fallback import page_is_blocked, parse_bing_results

RESULT_PREFIX = "OPENCODE_STEALTH_RESULT="


def emit(payload: dict) -> None:
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def selected_html(document: str, selector: str | None) -> str:
    if not selector:
        return document
    tree = lxml_html.fromstring(document)
    nodes = tree.cssselect(selector)
    return "\n".join(lxml_html.tostring(node, encoding="unicode") for node in nodes)


def main() -> int:
    request = json.loads(sys.stdin.read())
    timeout_seconds = max(int(request.get("timeout", 60)), 60)
    page = StealthyFetcher.fetch(
        request["url"],
        headless=True,
        real_chrome=True,
        solve_cloudflare=True,
        timeout=timeout_seconds * 1000,
        network_idle=False,
        retries=3,
    )
    document = page.body.decode("utf-8", errors="replace")
    title_node = page.css("title::text")
    title = title_node.get() if title_node else None
    if request.get("operation") == "search":
        results = parse_bing_results(document, min(max(int(request.get("max_results", 5)), 1), 10))
        # Valid organic nodes are stronger evidence than a marker word that may
        # legitimately occur inside one result.  Only classify an empty search
        # page as an interstitial.
        if not results and (page_is_blocked(document) or (title or "").strip().lower() == "just a moment..."):
            raise RuntimeError(
                "stealth browser remained blocked "
                f"(HTTP {page.status}, title={title!r}, reason=interstitial)"
            )
        if not results:
            raise RuntimeError("stealth browser returned no validated organic search results")
        emit(
            {
                "ok": True,
                "provider": "bing-browser-fallback",
                "via": "stealth",
                "results": results,
            }
        )
        return 0

    detected_block, block_reason = is_blocked(page.status, document)
    if detected_block or (title or "").strip().lower() == "just a moment...":
        raise RuntimeError(
            "stealth browser remained blocked "
            f"(HTTP {page.status}, title={title!r}, reason={block_reason or 'interstitial'})"
        )

    document = selected_html(document, request.get("css_selector"))
    markdown = trafilatura.extract(
        document,
        output_format="markdown",
        include_links=True,
        include_images=False,
        favor_recall=True,
    )
    if not markdown:
        markdown = page.get_all_text(separator="\n", strip=True)
    emit(
        {
            "ok": bool(markdown),
            "url": page.url,
            "title": title,
            "markdown": markdown or "",
            "via": "stealth",
        }
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit({"ok": False, "error": f"{type(error).__name__}: {error}"})
        raise SystemExit(1)
