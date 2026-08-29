"""Pure helpers for the bounded Bing browser-search fallback."""

from __future__ import annotations

import base64
import re
from urllib.parse import parse_qs, quote_plus, urlsplit, urlunsplit

from lxml import html as lxml_html

BLOCK_MARKERS = (
    "access denied",
    "captcha",
    "checking your browser",
    "consent required",
    "just a moment",
    "our systems have detected",
    "unusual traffic",
    "verify you are a human",
)
LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z]{2})?$")


def build_bing_search_url(query: str, language: str | None = None) -> str:
    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    if language:
        if not LANGUAGE_RE.fullmatch(language):
            raise ValueError("language must be a two- or three-letter code with an optional region")
        url += f"&setlang={quote_plus(language)}"
    return url


def page_is_blocked(document: str) -> bool:
    lowered = document.lower()
    return any(marker in lowered for marker in BLOCK_MARKERS)


def decode_bing_url(href: str) -> str | None:
    """Return a direct public candidate or decode Bing's known `u=a1...` wrapper."""

    parsed = urlsplit(href)
    if parsed.scheme in {"http", "https"} and parsed.hostname not in {
        "bing.com",
        "www.bing.com",
    }:
        return href
    if parsed.hostname not in {None, "bing.com", "www.bing.com"} or parsed.path != "/ck/a":
        return None
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if not encoded.startswith("a1"):
        return None
    payload = encoded[2:]
    try:
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    return decoded if urlsplit(decoded).scheme in {"http", "https"} else None


def structurally_safe_result_url(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        return None
    if hostname in {"bing.com", "www.bing.com"}:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))


def parse_bing_results(document: str, max_results: int = 10) -> list[dict]:
    tree = lxml_html.fromstring(document or "<html></html>")
    results: list[dict] = []
    seen: set[str] = set()
    for node in tree.cssselect("li.b_algo"):
        anchors = node.cssselect("h2 a[href]")
        if not anchors:
            continue
        anchor = anchors[0]
        target = structurally_safe_result_url(decode_bing_url(anchor.get("href", "")) or "")
        if not target or target in seen:
            continue
        title = " ".join(anchor.text_content().split())
        if not title:
            continue
        snippets = node.cssselect(".b_caption p, p")
        snippet = " ".join(snippets[0].text_content().split()) if snippets else ""
        results.append(
            {
                "title": title,
                "url": target,
                "content": snippet,
                "engine": "bing-browser-fallback",
                "engines": ["bing-browser-fallback"],
            }
        )
        seen.add(target)
        if len(results) >= max_results:
            break
    return results
