"""Crawl4AI fetch server for opencode — clean markdown + deep crawl + extraction.

Endpoints:
  GET  /healthz
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

HOST = os.environ.get("FETCH_HOST", "127.0.0.1")
PORT = int(os.environ.get("FETCH_PORT", "11235"))
STATIC_MIN_CHARS = 800  # below this in auto mode, escalate to browser
DEFAULT_MAX_CHARS = 40_000
ANTI_BOT_MARKERS = (
    "anti-bot protection",
    "access denied",
    "cf-challenge",
    "checking your browser",
    "cloudflare",
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
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=req.timeout,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        },
    ) as client:
        resp = await client.get(req.url)
        html = resp.text
        from crawl4ai.antibot_detector import is_blocked

        blocked, reason = is_blocked(resp.status_code, html)
        if blocked:
            raise AntiBotError(reason or f"anti-bot HTTP {resp.status_code}")
        resp.raise_for_status()
    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        include_links=True,
        include_images=False,
        favor_recall=True,
    )
    return {
        "url": str(resp.url),
        "title": None,
        "markdown": extracted or "",
        "via": "static",
    }


async def fetch_browser(req: FetchRequest) -> dict:
    from crawl4ai import CrawlerRunConfig, CacheMode

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
    return {
        "url": result.url,
        "title": (result.metadata or {}).get("title"),
        "markdown": md_of(result),
        "via": "browser",
    }


async def fetch_stealth(req: FetchRequest) -> dict:
    """Fetch a protected page with Scrapling's Cloudflare-aware browser.

    Scrapling is deliberately serialized because each request launches a real
    browser and Cloudflare challenges are CPU/memory intensive. The ordinary
    static and warm Crawl4AI paths remain concurrent.
    """

    async with _stealth_lock:
        logger.info("escalating to stealth browser url=%s", req.url)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(STEALTH_HELPER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        request_json = json.dumps(
            {
                "url": req.url,
                "css_selector": req.css_selector,
                "timeout": max(req.timeout, STEALTH_MIN_TIMEOUT_SECONDS),
            }
        ).encode()
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=request_json),
                timeout=max(req.timeout, STEALTH_MIN_TIMEOUT_SECONDS) + STEALTH_TIMEOUT_GRACE_SECONDS,
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
                f"{max(req.timeout, STEALTH_MIN_TIMEOUT_SECONDS) + STEALTH_TIMEOUT_GRACE_SECONDS}s"
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
        "stealth_available": stealth_version is not None,
        "stealth_version": stealth_version,
    }


@app.post("/fetch")
async def fetch(req: FetchRequest):
    try:
        if req.mode == "static":
            out = await fetch_static(req)
        elif req.mode == "browser":
            out = await fetch_browser(req)
        elif req.mode == "stealth":
            out = await fetch_stealth(req)
        else:  # auto
            try:
                out = await fetch_static(req)
                if len(out["markdown"]) < STATIC_MIN_CHARS:
                    out = await fetch_browser(req)
            except Exception as static_error:
                logger.debug("static fetch failed url=%s error=%s", req.url, static_error)
                try:
                    out = await fetch_browser(req)
                except Exception as browser_error:
                    if not is_antibot_error(static_error) and not is_antibot_error(browser_error):
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
