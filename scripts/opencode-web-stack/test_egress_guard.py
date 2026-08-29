import gzip
import os
import unittest
from unittest.mock import patch

import httpx

import egress_guard


PUBLIC_V4 = "93.184.216.34"


class CompressedStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content

    async def __aiter__(self):
        yield self.content


async def public_resolver(hostname: str, port: int):
    del hostname, port
    return [PUBLIC_V4]


class EgressGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_non_public_addresses_and_credentialed_urls(self):
        blocked = (
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://100.64.0.1/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[2002:7f00:1::]/",
            "http://[2001:0000:4136:e378:8000:63bf:3fff:fdd2]/",
            "http://[64:ff9b::7f00:1]/",
            "http://user:pass@example.com/",
            "http://localhost/",
            "http://example.com:8123/",
        )
        for url in blocked:
            with self.subTest(url=url), self.assertRaises(egress_guard.UnsafeUrlError):
                await egress_guard.resolve_public_target(url)

    async def test_rejects_hostname_when_any_dns_answer_is_private(self):
        async def mixed_resolver(hostname: str, port: int):
            del hostname, port
            return [PUBLIC_V4, "192.168.1.1"]

        with self.assertRaisesRegex(egress_guard.UnsafeUrlError, "outside the public internet"):
            await egress_guard.resolve_public_target("https://example.com/", mixed_resolver)

    async def test_pins_address_and_preserves_host_and_tls_sni(self):
        observed = {}

        async def handler(request: httpx.Request):
            observed["url"] = str(request.url)
            observed["host"] = request.headers["host"]
            observed["sni"] = request.extensions.get("sni_hostname")
            return httpx.Response(200, text="hello")

        response = await egress_guard.guarded_get(
            "https://example.com/docs?q=1",
            timeout=1,
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )
        self.assertEqual(response.text, "hello")
        self.assertEqual(observed["url"], f"https://{PUBLIC_V4}/docs?q=1")
        self.assertEqual(observed["host"], "example.com")
        self.assertEqual(observed["sni"], "example.com")

    async def test_revalidates_every_redirect_and_blocks_public_to_private(self):
        async def handler(request: httpx.Request):
            return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

        with self.assertRaises(egress_guard.UnsafeUrlError):
            await egress_guard.guarded_get(
                "https://example.com/",
                timeout=1,
                resolver=public_resolver,
                transport=httpx.MockTransport(handler),
            )

    async def test_rebinding_on_second_hop_is_rejected(self):
        calls = 0

        async def rebinding_resolver(hostname: str, port: int):
            nonlocal calls
            del hostname, port
            calls += 1
            return [PUBLIC_V4] if calls == 1 else ["127.0.0.1"]

        async def handler(request: httpx.Request):
            return httpx.Response(302, headers={"location": "/next"})

        with self.assertRaises(egress_guard.UnsafeUrlError):
            await egress_guard.guarded_get(
                "https://example.com/",
                timeout=1,
                resolver=rebinding_resolver,
                transport=httpx.MockTransport(handler),
            )
        self.assertEqual(calls, 2)

    async def test_https_to_http_redirect_is_rejected(self):
        async def handler(request: httpx.Request):
            return httpx.Response(302, headers={"location": "http://example.org/"})

        with self.assertRaisesRegex(egress_guard.UnsafeUrlError, "HTTPS-to-HTTP"):
            await egress_guard.guarded_get(
                "https://example.com/",
                timeout=1,
                resolver=public_resolver,
                transport=httpx.MockTransport(handler),
            )

    async def test_automatic_path_disables_environment_proxies(self):
        async def handler(request: httpx.Request):
            return httpx.Response(200, text="ok")

        with (
            patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:9999"}),
            patch("egress_guard.httpx.AsyncClient", wraps=httpx.AsyncClient) as client,
        ):
            await egress_guard.guarded_get(
                "https://example.com/",
                timeout=1,
                resolver=public_resolver,
                transport=httpx.MockTransport(handler),
            )
        self.assertIs(client.call_args.kwargs["trust_env"], False)

    async def test_real_streaming_path_decodes_compressed_body_once(self):
        expected = "compressed public page"

        async def handler(request: httpx.Request):
            return httpx.Response(
                200,
                headers={
                    "content-encoding": "gzip",
                    "content-type": "text/plain; charset=utf-8",
                },
                stream=CompressedStream(gzip.compress(expected.encode())),
            )

        response = await egress_guard.guarded_get(
            "https://example.com/",
            timeout=1,
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )
        self.assertEqual(response.text, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
