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
    def result(self):
        return {
            "title": "Example",
            "url": "https://example.com/",
            "content": "Example result",
            "engine": "bing-browser-fallback",
            "engines": ["bing-browser-fallback"],
        }

    async def test_browser_success_never_invokes_stealth(self):
        req = server.SearchFallbackRequest(query="example", max_results=3)
        with (
            patch.object(server, "search_browser", AsyncMock(return_value=[self.result()])) as browser,
            patch.object(server, "search_stealth", AsyncMock()) as stealth,
            patch.object(server, "resolve_public_target", AsyncMock()),
        ):
            result = await server.search_fallback(req)

        self.assertTrue(result["ok"])
        self.assertEqual(result["via"], "browser")
        self.assertEqual(result["provider"], "bing-browser-fallback")
        browser.assert_awaited_once()
        stealth.assert_not_awaited()

    async def test_stealth_is_one_bounded_second_attempt(self):
        req = server.SearchFallbackRequest(query="example")
        with (
            patch.object(server, "search_browser", AsyncMock(return_value=[])) as browser,
            patch.object(server, "search_stealth", AsyncMock(return_value=[self.result()])) as stealth,
            patch.object(server, "resolve_public_target", AsyncMock()),
        ):
            result = await server.search_fallback(req)

        self.assertTrue(result["ok"])
        self.assertEqual(result["via"], "stealth")
        browser.assert_awaited_once()
        stealth.assert_awaited_once()

    async def test_both_routes_fail_honestly(self):
        req = server.SearchFallbackRequest(query="example")
        with (
            patch.object(server, "search_browser", AsyncMock(side_effect=server.AntiBotError("CAPTCHA"))),
            patch.object(server, "search_stealth", AsyncMock(return_value=[])),
        ):
            result = await server.search_fallback(req)

        self.assertFalse(result["ok"])
        self.assertEqual(result["results"], [])
        self.assertIn("CAPTCHA", result["error"])
        self.assertIn("no organic results", result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
