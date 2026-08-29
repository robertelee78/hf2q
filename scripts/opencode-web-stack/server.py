"""Crawl4AI fetch server for opencode — clean markdown + deep crawl + extraction.

Endpoints:
  GET  /healthz
  POST /search-fallback {query, language?, max_results?}
  POST /fetch     {url, mode=auto|static|browser|stealth, css_selector?, max_chars?, timeout?}
  POST /crawl     {url, max_depth?, max_pages?, allowed_domains?, blocked_domains?,
                   include_external?, query?, max_chars?}
  POST /extract   {url, strategy=json_css|cosine, schema?, query?, max_chars?}

Modes:
  static  — httpx + trafilatura (fast, no JS)
  browser — headless Chromium via Crawl4AI (JS-rendered, fit_markdown)
  stealth — Patchright via Scrapling (Cloudflare/Turnstile-aware)
  auto    — static, then browser, then stealth only for detected anti-bot blocks
"""
import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import trafilatura
from fastapi import FastAPI
from pydantic import BaseModel, Field

from egress_guard import UnsafeUrlError, guarded_get, resolve_public_target
from search_fallback import build_bing_search_url, page_is_blocked, parse_bing_results

HOST = os.environ.get("FETCH_HOST", "127.0.0.1")
PORT = int(os.environ.get("FETCH_PORT", "11235"))
STATIC_MIN_CHARS = 800  # below this in auto mode, escalate to browser
DEFAULT_MAX_CHARS = 40_000
ANTI_BOT_MARKERS = (
    "anti-bot protection",
    "access denied",
    "captcha",
    "cf-challenge",
    "checking your browser",
    "cloudflare",
    "consent interstitial",
    "just a moment",
    "performing security verification",
    "turnstile",
)

logging.basicConfig(
    level=getattr(logging, os.environ.get("FETCH_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("opencode-fetch")

_crawler = None
_crawler_lock = asyncio.Lock()
_stealth_lock = asyncio.Lock()
STEALTH_HELPER = Path(__file__).with_name("stealth_fetch.py")
STEALTH_RESULT_PREFIX = "OPENCODE_STEALTH_RESULT="
STEALTH_MIN_TIMEOUT_SECONDS = 60
STEALTH_TIMEOUT_GRACE_SECONDS = 15


class AntiBotError(RuntimeError):
    """A fetch reached an anti-bot interstitial instead of the requested page."""


def is_antibot_error(error: BaseException | str) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in ANTI_BOT_MARKERS)


async def get_crawler():
    """Lazy-start one warm AsyncWebCrawler and reuse the browser process."""
    global _crawler
    async with _crawler_lock:
        if _crawler is None:
            from crawl4ai import AsyncWebCrawler, BrowserConfig

            cfg = BrowserConfig(headless=True, verbose=False)
            _crawler = AsyncWebCrawler(config=cfg)
            await _crawler.start()
    return _crawler


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _crawler is not None:
        await _crawler.close()


app = FastAPI(title="opencode-fetch", lifespan=lifespan)


class FetchRequest(BaseModel):
    url: str
    mode: str = Field(default="auto", pattern="^(auto|static|browser|stealth)$")
    css_selector: str | None = None
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, le=200_000)
    timeout: int = Field(default=75, le=120)
    public_only: bool = False


class SearchFallbackRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    language: str | None = Field(default=None, pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z]{2})?$")
    max_results: int = Field(default=5, ge=1, le=10)


class CrawlRequest(BaseModel):
    url: str
    max_depth: int = Field(default=2, ge=1, le=10)
    max_pages: int = Field(default=20, ge=1, le=200)
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    include_external: bool = False
    query: str | None = None
    max_chars: int = Field(default=12_000, le=200_000)


class ExtractRequest(BaseModel):
    url: str
    strategy: str = Field(default="json_css", pattern="^(json_css|cosine)$")
    extraction_schema: dict | None = Field(default=None, alias="schema")
    query: str | None = None
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, le=200_000)
    timeout: int = Field(default=45, le=120)


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def md_of(result) -> str:
    """Extract fit (clean) markdown from a CrawlResult, falling back to raw."""
    if not result.markdown:
        return ""
    return result.markdown.fit_markdown or result.markdown.raw_markdown or ""


async def fetch_static(req: FetchRequest) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    }
    if req.public_only:
        resp = await guarded_get(req.url, timeout=req.timeout, headers=headers)
        response_url = resp.url
        status_code = resp.status_code
        html = resp.text
    else:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=req.timeout,
            headers=headers,
        ) as client:
            response = await client.get(req.url)
            response_url = str(response.url)
            status_code = response.status_code
            html = response.text

    from crawl4ai.antibot_detector import is_blocked

    blocked, reason = is_blocked(status_code, html)
    if blocked:
        raise AntiBotError(reason or f"anti-bot HTTP {status_code}")
    if status_code >= 400:
        raise RuntimeError(f"HTTP {status_code}")
    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        include_links=True,
        include_images=False,
        favor_recall=True,
    )
    return {
        "url": response_url,
        "title": None,
        "markdown": extracted or "",
        "via": "static",
    }


async def fetch_browser(req: FetchRequest) -> dict:
    from crawl4ai import CrawlerRunConfig, CacheMode
    from crawl4ai.antibot_detector import is_blocked

    crawler = await get_crawler()
    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector=req.css_selector,
        page_timeout=req.timeout * 1000,
        word_count_threshold=10,
    )
    result = await crawler.arun(url=req.url, config=cfg)
    if not result.success:
        error = result.error_message or "crawl failed"
        if is_antibot_error(error):
            raise AntiBotError(error)
        raise RuntimeError(error)
    blocked, reason = is_blocked(getattr(result, "status_code", 200) or 200, result.html or "")
    if blocked:
        raise AntiBotError(reason or "browser reached an anti-bot interstitial")
    return {
        "url": result.url,
        "title": (result.metadata or {}).get("title"),
        "markdown": md_of(result),
        "via": "browser",
    }


async def run_stealth_helper(request: dict, timeout: int) -> dict:
    async with _stealth_lock:
        logger.info("escalating to stealth browser url=%s", request["url"])
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(STEALTH_HELPER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        request["timeout"] = max(timeout, STEALTH_MIN_TIMEOUT_SECONDS)
        request_json = json.dumps(request).encode()
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=request_json),
                timeout=max(timeout, STEALTH_MIN_TIMEOUT_SECONDS) + STEALTH_TIMEOUT_GRACE_SECONDS,
            )
        except TimeoutError as error:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
            raise TimeoutError(
                "stealth browser exceeded "
                f"{max(timeout, STEALTH_MIN_TIMEOUT_SECONDS) + STEALTH_TIMEOUT_GRACE_SECONDS}s"
            ) from error

        stderr_text = stderr.decode(errors="replace").strip()
        if stderr_text:
            logger.debug("stealth helper stderr=%s", stderr_text[-4_000:])
        payload_line = next(
            (
                line[len(STEALTH_RESULT_PREFIX) :]
                for line in reversed(stdout.decode(errors="replace").splitlines())
                if line.startswith(STEALTH_RESULT_PREFIX)
            ),
            None,
        )
        if payload_line is None:
            detail = stderr_text.splitlines()[-1] if stderr_text else "no result payload"
            raise RuntimeError(f"stealth helper failed: {detail}")
        payload = json.loads(payload_line)
        if process.returncode != 0 or not payload.get("ok"):
            raise RuntimeError(payload.get("error") or f"stealth helper exited {process.returncode}")
        return payload


async def fetch_stealth(req: FetchRequest) -> dict:
    """Fetch a protected page with one serialized Scrapling worker."""

    return await run_stealth_helper(
        {
            "operation": "fetch",
            "url": req.url,
            "css_selector": req.css_selector,
        },
        req.timeout,
    )


async def search_browser(url: str, max_results: int) -> list[dict]:
    from crawl4ai import CacheMode, CrawlerRunConfig

    crawler = await get_crawler()
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=45_000,
        word_count_threshold=0,
    )
    result = await crawler.arun(url=url, config=config)
    if not result.success:
        raise RuntimeError(result.error_message or "browser search failed")
    results = parse_bing_results(result.html, max_results)
    if not results and page_is_blocked(result.html):
        raise AntiBotError("browser search reached a CAPTCHA or consent interstitial")
    return results


async def search_stealth(url: str, max_results: int) -> list[dict]:
    payload = await run_stealth_helper(
        {"operation": "search", "url": url, "max_results": max_results},
        75,
    )
    return payload.get("results", [])


async def validated_search_results(results: list[dict]) -> list[dict]:
    validated = []
    for result in results:
        try:
            await asyncio.wait_for(resolve_public_target(result["url"]), timeout=5)
        except (KeyError, UnsafeUrlError, OSError) as error:
            logger.warning("discarding unsafe search result url=%r error=%s", result.get("url"), error)
            continue
        validated.append(result)
    return validated


@app.get("/healthz")
async def healthz():
    try:
        from importlib.metadata import version

        stealth_version = version("scrapling")
    except Exception:
        stealth_version = None
    return {
        "ok": True,
        "browser_warm": _crawler is not None,
        "stealth_installed": stealth_version is not None,
        "stealth_version": stealth_version,
    }


@app.post("/search-fallback")
async def search_fallback(req: SearchFallbackRequest):
    """One bounded, fixed-origin browser discovery attempt after SearXNG fails."""

    url = build_bing_search_url(req.query, req.language)
    attempts = []
    for route, search in (("browser", search_browser), ("stealth", search_stealth)):
        try:
            results = await validated_search_results(await search(url, req.max_results))
            if results:
                return {
                    "ok": True,
                    "provider": "bing-browser-fallback",
                    "via": route,
                    "results": results,
                }
            attempts.append(f"{route}: no organic results")
        except Exception as error:
            attempts.append(f"{route}: {type(error).__name__}: {error}")
    return {
        "ok": False,
        "provider": "bing-browser-fallback",
        "results": [],
        "error": "; ".join(attempts),
    }


@app.post("/fetch")
async def fetch(req: FetchRequest):
    try:
        if req.public_only and req.mode not in {"auto", "static"}:
            raise UnsafeUrlError("public_only automatic reads are static-only")
        if req.public_only or req.mode == "static":
            out = await fetch_static(req)
        elif req.mode == "browser":
            out = await fetch_browser(req)
        elif req.mode == "stealth":
            out = await fetch_stealth(req)
        else:  # auto
            static_error = None
            try:
                out = await fetch_static(req)
            except Exception as error:
                static_error = error
                logger.debug("static fetch failed url=%s error=%s", req.url, static_error)

            if static_error is not None or len(out["markdown"]) < STATIC_MIN_CHARS:
                try:
                    out = await fetch_browser(req)
                except Exception as browser_error:
                    if not is_antibot_error(static_error or "") and not is_antibot_error(browser_error):
                        raise
                    out = await fetch_stealth(req)
        md, truncated = truncate(out["markdown"], req.max_chars)
        return {
            "ok": bool(md),
            "url": out["url"],
            "title": out["title"],
            "markdown": md,
            "via": out["via"],
            "truncated": truncated,
        }
    except Exception as e:
        return {"ok": False, "url": req.url, "error": f"{type(e).__name__}: {e}"}


@app.post("/crawl")
async def crawl(req: CrawlRequest):
    """Deep-crawl a site via BFS, optionally relevance-filtered by a query."""
    try:
        from crawl4ai import CacheMode, CrawlerRunConfig
        from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
        from crawl4ai.deep_crawling.filters import (
            ContentRelevanceFilter,
            DomainFilter,
            FilterChain,
            URLPatternFilter,
        )

        crawler = await get_crawler()

        filters = []
        if req.allowed_domains or req.blocked_domains:
            filters.append(
                DomainFilter(
                    allowed_domains=req.allowed_domains,
                    blocked_domains=req.blocked_domains,
                )
            )
        if req.query:
            filters.append(ContentRelevanceFilter(query=req.query, threshold=0.5))

        strategy = BFSDeepCrawlStrategy(
            max_depth=req.max_depth,
            max_pages=req.max_pages,
            include_external=req.include_external,
            filter_chain=FilterChain(filters=filters),
        )

        cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=45_000,
            word_count_threshold=10,
        )

        results = await strategy.arun(start_url=req.url, crawler=crawler, config=cfg)

        pages = []
        for result in results:
            md = md_of(result)
            if not md:
                continue
            md, truncated = truncate(md, req.max_chars)
            meta = result.metadata or {}
            pages.append(
                {
                    "url": result.url,
                    "status": getattr(result, "status_code", None),
                    "title": meta.get("title"),
                    "depth": meta.get("depth"),
                    "markdown": md,
                    "truncated": truncated,
                }
            )
            if len(pages) >= req.max_pages:
                break

        return {"ok": True, "crawled": len(pages), "pages": pages}
    except Exception as e:
        return {"ok": False, "url": req.url, "error": f"{type(e).__name__}: {e}"}


@app.post("/extract")
async def extract(req: ExtractRequest):
    """Structured extraction: JSON from a CSS schema, or semantic clusters."""
    try:
        from crawl4ai import CacheMode, CrawlerRunConfig
        from crawl4ai.extraction_strategy import (
            CosineStrategy,
            JsonCssExtractionStrategy,
        )

        if req.strategy == "json_css":
            if not req.extraction_schema:
                return {
                    "ok": False,
                    "url": req.url,
                    "error": "json_css requires a 'schema' (JsonCssExtractionStrategy schema)",
                }
            extraction = JsonCssExtractionStrategy(schema=req.extraction_schema)
        else:  # cosine
            extraction = CosineStrategy(semantic_filter=req.query)

        crawler = await get_crawler()
        cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction,
            page_timeout=req.timeout * 1000,
            word_count_threshold=10,
        )
        result = await crawler.arun(url=req.url, config=cfg)
        if not result.success:
            raise RuntimeError(result.error_message or "crawl failed")

        data = result.extracted_content
        return {
            "ok": True,
            "url": result.url,
            "strategy": req.strategy,
            "data": data,
        }
    except Exception as e:
        return {"ok": False, "url": req.url, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
