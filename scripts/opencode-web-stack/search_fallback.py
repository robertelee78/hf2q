"""Pure helpers for the bounded, provider-diverse search fallback."""

from __future__ import annotations

import base64
import re
from urllib.parse import parse_qs, quote_plus, urlsplit, urlunsplit

from lxml import etree
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
QUERY_STOP_WORDS = {
    "about",
    "company",
    "current",
    "does",
    "for",
    "from",
    "how",
    "into",
    "latest",
    "please",
    "tell",
    "that",
    "the",
    "their",
    "this",
    "today",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "wrote",
}
SEARCH_PROVIDER_HOSTS = {
    "bing.com",
    "www.bing.com",
    "search.brave.com",
}


def significant_query_terms(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9.+#-]*", query.lower())
    return list(
        dict.fromkeys(
            term for term in tokens if len(term) >= 3 and term not in QUERY_STOP_WORDS
        )
    )


def focused_query(query: str) -> str:
    """Preserve natural-language intent while requiring its identifying terms."""

    terms = significant_query_terms(query)
    if not terms:
        return query
    anchors = " ".join(f'"{term}"' for term in terms[:6])
    return f"{query} {anchors}"


def _term_variants(term: str) -> set[str]:
    """Return conservative singular/plural variants without a language dependency."""

    variants = {term}
    if len(term) > 4 and term.endswith("ies"):
        variants.add(f"{term[:-3]}y")
    elif term.endswith(("ches", "shes", "sses", "xes", "zes")):
        variants.add(term[:-2])
    elif len(term) > 3 and term.endswith("es"):
        variants.add(term[:-1])
    elif (
        len(term) > 3
        and term.endswith("s")
        and not term.endswith(("ss", "is", "us"))
        and term not in {"news", "series", "species"}
    ):
        variants.add(term[:-1])
    elif term.endswith("y") and len(term) > 1 and term[-2] not in "aeiou":
        variants.add(f"{term[:-1]}ies")
    elif term.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(f"{term}es")
    elif not term.endswith(("is", "us")):
        variants.add(f"{term}s")
    return variants


def _evidence_contains_term(evidence: str, term: str) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", evidence)
        is not None
        for variant in _term_variants(term)
    )


def result_looks_relevant(query: str, result: dict) -> bool:
    terms = significant_query_terms(query)
    if not terms:
        return False
    evidence = " ".join(
        str(result.get(field) or "") for field in ("title", "url", "content")
    ).lower()
    matched = sum(_evidence_contains_term(evidence, term) for term in terms)
    return matched == len(terms) or (len(terms) >= 3 and matched * 3 >= len(terms) * 2)


def filter_relevant_results(
    query: str, results: list[dict], max_results: int
) -> list[dict]:
    return [result for result in results if result_looks_relevant(query, result)][:max_results]


def build_bing_search_url(query: str, language: str | None = None) -> str:
    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    if language:
        if not LANGUAGE_RE.fullmatch(language):
            raise ValueError("language must be a two- or three-letter code with an optional region")
        url += f"&setlang={quote_plus(language)}"
    return url


def build_bing_rss_search_url(query: str, language: str | None = None) -> str:
    url = f"https://www.bing.com/search?q={quote_plus(query)}&format=rss"
    if language:
        if not LANGUAGE_RE.fullmatch(language):
            raise ValueError("language must be a two- or three-letter code with an optional region")
        url += f"&setlang={quote_plus(language)}"
    return url


def build_brave_search_url(query: str, language: str | None = None) -> str:
    url = f"https://search.brave.com/search?q={quote_plus(query)}&source=web&spellcheck=0"
    if language:
        if not LANGUAGE_RE.fullmatch(language):
            raise ValueError("language must be a two- or three-letter code with an optional region")
        url += f"&lang={quote_plus(language)}"
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
    if hostname in SEARCH_PROVIDER_HOSTS:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))


def _organic_result(title: str, target: str, content: str, provider: str) -> dict:
    return {
        "title": " ".join(title.split()),
        "url": target,
        "content": " ".join(content.split()),
        "engine": provider,
        "engines": [provider],
    }


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
        results.append(_organic_result(title, target, snippet, "bing-browser-fallback"))
        seen.add(target)
        if len(results) >= max_results:
            break
    return results


def parse_bing_rss_results(document: str, max_results: int = 10) -> list[dict]:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    try:
        tree = etree.fromstring((document or "").encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError:
        return []

    results: list[dict] = []
    seen: set[str] = set()
    for item in tree.xpath(".//item"):
        title = (item.findtext("title") or "").strip()
        target = structurally_safe_result_url((item.findtext("link") or "").strip())
        content = item.findtext("description") or ""
        if not title or not target or target in seen:
            continue
        results.append(_organic_result(title, target, content, "bing-rss-fallback"))
        seen.add(target)
        if len(results) >= max_results:
            break
    return results


def parse_brave_results(document: str, max_results: int = 10) -> list[dict]:
    tree = lxml_html.fromstring(document or "<html></html>")
    results: list[dict] = []
    seen: set[str] = set()
    for node in tree.cssselect('.snippet[data-type="web"]'):
        anchors = node.cssselect("a[href]")
        title_nodes = node.cssselect(".search-snippet-title, .title")
        if not anchors or not title_nodes:
            continue
        target = structurally_safe_result_url(anchors[0].get("href", ""))
        title = " ".join(title_nodes[0].text_content().split())
        if not title or not target or target in seen:
            continue
        snippets = node.cssselect(".generic-snippet, .snippet-description")
        content = " ".join(snippets[0].text_content().split()) if snippets else ""
        results.append(_organic_result(title, target, content, "brave-search-fallback"))
        seen.add(target)
        if len(results) >= max_results:
            break
    return results
