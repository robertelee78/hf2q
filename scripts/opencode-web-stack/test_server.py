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


if __name__ == "__main__":
    unittest.main(verbosity=2)
