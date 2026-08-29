import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import server


class FetchRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_escalates_antibot_failure_to_stealth(self):
        req = server.FetchRequest(url="https://protected.example", mode="auto")
        with (
            patch.object(
                server,
                "fetch_static",
                AsyncMock(side_effect=server.AntiBotError("Cloudflare challenge")),
            ),
            patch.object(
                server,
                "fetch_browser",
                AsyncMock(side_effect=server.AntiBotError("Cloudflare JS challenge")),
            ),
            patch.object(
                server,
                "fetch_stealth",
                AsyncMock(
                    return_value={
                        "url": req.url,
                        "title": "Protected",
                        "markdown": "real content",
                        "via": "stealth",
                    }
                ),
            ) as stealth,
        ):
            result = await server.fetch(req)

        self.assertTrue(result["ok"])
        self.assertEqual(result["via"], "stealth")
        stealth.assert_awaited_once_with(req)

    async def test_auto_does_not_mask_unrelated_browser_failure(self):
        req = server.FetchRequest(url="https://broken.example", mode="auto")
        with (
            patch.object(
                server,
                "fetch_static",
                AsyncMock(side_effect=RuntimeError("connection reset")),
            ),
            patch.object(
                server,
                "fetch_browser",
                AsyncMock(side_effect=RuntimeError("DNS failure")),
            ),
            patch.object(server, "fetch_stealth", AsyncMock()) as stealth,
        ):
            result = await server.fetch(req)

        self.assertFalse(result["ok"])
        self.assertIn("DNS failure", result["error"])
        stealth.assert_not_awaited()

    async def test_short_static_then_blocked_browser_escalates_without_retry(self):
        req = server.FetchRequest(url="https://protected.example", mode="auto")
        static_result = {
            "url": req.url,
            "title": None,
            "markdown": "short",
            "via": "static",
        }
        stealth_result = {
            "url": req.url,
            "title": "Protected",
            "markdown": "real content",
            "via": "stealth",
        }
        with (
            patch.object(server, "fetch_static", AsyncMock(return_value=static_result)),
            patch.object(
                server,
                "fetch_browser",
                AsyncMock(side_effect=server.AntiBotError("CAPTCHA")),
            ) as browser,
            patch.object(server, "fetch_stealth", AsyncMock(return_value=stealth_result)) as stealth,
        ):
            result = await server.fetch(req)

        self.assertTrue(result["ok"])
        self.assertEqual(result["via"], "stealth")
        browser.assert_awaited_once_with(req)
        stealth.assert_awaited_once_with(req)

    async def test_explicit_stealth_mode_routes_directly(self):
        req = server.FetchRequest(url="https://protected.example", mode="stealth")
        expected = {
            "url": req.url,
            "title": "Protected",
            "markdown": "real content",
            "via": "stealth",
        }
        with patch.object(server, "fetch_stealth", AsyncMock(return_value=expected)) as stealth:
            result = await server.fetch(req)

        self.assertEqual(result["via"], "stealth")
        stealth.assert_awaited_once_with(req)

    def test_antibot_classifier(self):
        self.assertTrue(server.is_antibot_error("Blocked by anti-bot protection: Cloudflare JS challenge"))
        self.assertTrue(server.is_antibot_error("Just a moment..."))
        self.assertFalse(server.is_antibot_error("DNS lookup failed"))

    async def test_stealth_worker_has_a_hard_wall_clock_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "slow_helper.py"
            helper.write_text("import sys, time\nsys.stdin.read()\ntime.sleep(30)\n")
            req = server.FetchRequest(url="https://protected.example", mode="stealth", timeout=1)
            started = time.monotonic()
            with (
                patch.object(server, "STEALTH_HELPER", helper),
                patch.object(server, "STEALTH_MIN_TIMEOUT_SECONDS", 1),
                patch.object(server, "STEALTH_TIMEOUT_GRACE_SECONDS", 0),
            ):
                result = await server.fetch(req)

        self.assertFalse(result["ok"])
        self.assertIn("stealth browser exceeded 1s", result["error"])
        self.assertLess(time.monotonic() - started, 5)

    async def test_public_only_automatic_read_is_static_only(self):
        req = server.FetchRequest(
            url="https://public.example",
            mode="auto",
            public_only=True,
        )
        expected = {
            "url": req.url,
            "title": None,
            "markdown": "public content",
            "via": "static",
        }
        with (
            patch.object(server, "fetch_static", AsyncMock(return_value=expected)) as static,
            patch.object(server, "fetch_browser", AsyncMock()) as browser,
        ):
            result = await server.fetch(req)

        self.assertTrue(result["ok"])
        static.assert_awaited_once_with(req)
        browser.assert_not_awaited()

    async def test_public_only_rejects_explicit_browser_mode(self):
        req = server.FetchRequest(
            url="https://public.example",
            mode="browser",
            public_only=True,
        )
        result = await server.fetch(req)
        self.assertFalse(result["ok"])
        self.assertIn("static-only", result["error"])


class SearchFallbackRoutingTests(unittest.IsolatedAsyncioTestCase):
    def result(self, provider="bing-browser-fallback"):
        return {
            "title": "Example",
            "url": "https://example.com/",
            "content": "Example result",
            "engine": provider,
            "engines": [provider],
        }

    async def test_brave_success_never_invokes_bing_routes(self):
        req = server.SearchFallbackRequest(query="example", max_results=3)
        with (
            patch.object(
                server,
                "search_static",
                AsyncMock(return_value=[self.result("brave-search-fallback")]),
            ) as static,
            patch.object(server, "search_browser", AsyncMock()) as browser,
            patch.object(server, "search_stealth", AsyncMock()) as stealth,
            patch.object(server, "resolve_public_target", AsyncMock()),
        ):
            result = await server.search_fallback(req)

        self.assertTrue(result["ok"])
        self.assertEqual(result["via"], "guarded-static")
        self.assertEqual(result["provider"], "brave-search-fallback")
        static.assert_awaited_once()
        browser.assert_not_awaited()
        stealth.assert_not_awaited()

    async def test_irrelevant_brave_continues_to_relevant_bing_rss(self):
        req = server.SearchFallbackRequest(query="what is the price of gold today")
        junk = {
            "title": "Price.com: Cash Back and Coupons",
            "url": "https://price.com/",
            "content": "Compare prices for popular brands.",
            "engine": "brave-search-fallback",
            "engines": ["brave-search-fallback"],
        }
        gold = {
            "title": "Gold Price Today",
            "url": "https://www.kitco.com/charts/gold",
            "content": "Live gold price per ounce.",
            "engine": "bing-rss-fallback",
            "engines": ["bing-rss-fallback"],
        }
        with (
            patch.object(
                server,
                "search_static",
                AsyncMock(side_effect=[[junk], [gold]]),
            ) as static,
            patch.object(server, "search_browser", AsyncMock()) as browser,
            patch.object(server, "search_stealth", AsyncMock()) as stealth,
            patch.object(server, "resolve_public_target", AsyncMock()),
        ):
            result = await server.search_fallback(req)

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "bing-rss-fallback")
        self.assertEqual(static.await_count, 2)
        browser.assert_not_awaited()
        stealth.assert_not_awaited()

    async def test_stealth_is_one_bounded_final_attempt(self):
        req = server.SearchFallbackRequest(query="example")
        with (
            patch.object(server, "search_static", AsyncMock(return_value=[])) as static,
            patch.object(server, "search_browser", AsyncMock(return_value=[])) as browser,
            patch.object(server, "search_stealth", AsyncMock(return_value=[self.result()])) as stealth,
            patch.object(server, "resolve_public_target", AsyncMock()),
        ):
            result = await server.search_fallback(req)

        self.assertTrue(result["ok"])
        self.assertEqual(result["via"], "stealth")
        self.assertEqual(static.await_count, 2)
        browser.assert_awaited_once()
        stealth.assert_awaited_once()

    async def test_static_route_timeout_advances_to_next_provider(self):
        req = server.SearchFallbackRequest(query="example")
        calls = 0

        async def static_route(*_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(1)
            return [self.result("bing-rss-fallback")]

        with (
            patch.object(server, "search_static", static_route),
            patch.object(server, "STATIC_SEARCH_ROUTE_TIMEOUT_SECONDS", 0.01),
            patch.object(server, "search_browser", AsyncMock()) as browser,
            patch.object(server, "resolve_public_target", AsyncMock()),
        ):
            result = await server.search_fallback(req)

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "bing-rss-fallback")
        self.assertIn("TimeoutError", result["attempts"][0])
        browser.assert_not_awaited()

    async def test_browser_route_timeout_advances_to_stealth(self):
        req = server.SearchFallbackRequest(query="example")

        async def slow_browser(*_args):
            await asyncio.sleep(1)
            return []

        with (
            patch.object(server, "search_static", AsyncMock(return_value=[])),
            patch.object(server, "search_browser", slow_browser),
            patch.object(server, "BROWSER_SEARCH_ROUTE_TIMEOUT_SECONDS", 0.01),
            patch.object(
                server,
                "search_stealth",
                AsyncMock(return_value=[self.result()]),
            ) as stealth,
            patch.object(server, "resolve_public_target", AsyncMock()),
        ):
            result = await server.search_fallback(req)

        self.assertTrue(result["ok"])
        self.assertEqual(result["via"], "stealth")
        self.assertIn("TimeoutError", result["attempts"][-1])
        stealth.assert_awaited_once()

    async def test_all_routes_fail_honestly(self):
        req = server.SearchFallbackRequest(query="example")
        with (
            patch.object(
                server,
                "search_static",
                AsyncMock(side_effect=[server.AntiBotError("blocked"), []]),
            ),
            patch.object(server, "search_browser", AsyncMock(side_effect=server.AntiBotError("CAPTCHA"))),
            patch.object(server, "search_stealth", AsyncMock(return_value=[])),
        ):
            result = await server.search_fallback(req)

        self.assertFalse(result["ok"])
        self.assertEqual(result["results"], [])
        self.assertIn("CAPTCHA", result["error"])
        self.assertIn("no query-relevant organic results", result["error"])
        self.assertEqual(result["provider"], "multi-provider-fallback")

    async def test_exact_laptop_junk_cannot_succeed_on_any_route(self):
        req = server.SearchFallbackRequest(
            query="what is the price of gold today", max_results=3
        )
        junk = [
            {
                "title": "Price.com: Save with Cash Back, Coupons & Price Comparison",
                "url": "https://price.com/",
                "content": "Offers for more than 100,000 brands.",
                "engine": "bing-browser-fallback",
                "engines": ["bing-browser-fallback"],
            },
            {
                "title": "Home - Price Industries",
                "url": "https://priceindustries.com/",
                "content": "A market leader in supplying air distribution products.",
                "engine": "bing-browser-fallback",
                "engines": ["bing-browser-fallback"],
            },
            {
                "title": "Priceline.com - Hotels, Flights and Rental Cars",
                "url": "https://www.priceline.com/",
                "content": "Members get our best travel price.",
                "engine": "bing-browser-fallback",
                "engines": ["bing-browser-fallback"],
            },
        ]
        with (
            patch.object(server, "search_static", AsyncMock(return_value=junk)),
            patch.object(server, "search_browser", AsyncMock(return_value=junk)),
            patch.object(server, "search_stealth", AsyncMock(return_value=junk)),
            patch.object(server, "resolve_public_target", AsyncMock()),
        ):
            result = await server.search_fallback(req)

        self.assertFalse(result["ok"])
        self.assertEqual(result["results"], [])
        self.assertEqual(result["provider"], "multi-provider-fallback")

    async def test_service_removes_irrelevant_siblings_before_success(self):
        req = server.SearchFallbackRequest(
            query="what is the price of gold today", max_results=3
        )
        junk = {
            "title": "Price.com: Cash Back and Coupons",
            "url": "https://price.com/",
            "content": "Compare prices for popular brands.",
            "engine": "brave-search-fallback",
            "engines": ["brave-search-fallback"],
        }
        gold = {
            "title": "Gold Price Today",
            "url": "https://www.kitco.com/charts/gold",
            "content": "Live gold price per ounce.",
            "engine": "brave-search-fallback",
            "engines": ["brave-search-fallback"],
        }
        with (
            patch.object(
                server, "search_static", AsyncMock(return_value=[junk, gold])
            ),
            patch.object(server, "search_browser", AsyncMock()) as browser,
            patch.object(server, "resolve_public_target", AsyncMock()),
        ):
            result = await server.search_fallback(req)

        self.assertTrue(result["ok"])
        self.assertEqual([item["url"] for item in result["results"]], [gold["url"]])
        browser.assert_not_awaited()

    async def test_result_url_validation_timeout_discards_candidate(self):
        with patch.object(
            server,
            "resolve_public_target",
            AsyncMock(side_effect=TimeoutError("DNS validation timed out")),
        ):
            results = await server.validated_search_results([self.result()])
        self.assertEqual(results, [])

    def test_fallback_budget_fits_plugin_deadline(self):
        self.assertLess(server.SEARCH_FALLBACK_WORST_CASE_SECONDS, 150)


if __name__ == "__main__":
    unittest.main(verbosity=2)
