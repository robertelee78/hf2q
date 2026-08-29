"""Fail-closed egress validation for automatically discovered web URLs.

Search results are untrusted input.  This module validates every DNS answer,
pins the connection to a validated address, preserves TLS SNI/Host for the
original name, disables environment proxies, and repeats the process for each
redirect hop.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

import httpx

Resolver = Callable[[str, int], Awaitable[Iterable[str]]]
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
DEFAULT_MAX_BYTES = 8 * 1024 * 1024


class UnsafeUrlError(ValueError):
    """An automatic request target crossed the public-web boundary."""


@dataclass(frozen=True)
class PublicTarget:
    logical_url: str
    hostname: str
    port: int
    address: str
    pinned_url: str
    host_header: str


@dataclass(frozen=True)
class GuardedResponse:
    status_code: int
    url: str
    text: str


def _globally_routable(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address):
        if parsed.ipv4_mapped:
            parsed = parsed.ipv4_mapped
        elif parsed.sixtofour is not None or parsed.teredo is not None:
            return False
        elif parsed in ipaddress.ip_network("64:ff9b::/96") or parsed in ipaddress.ip_network("64:ff9b:1::/48"):
            return False
    return parsed.is_global


def _parse_url(url: str) -> tuple[SplitResult, str, int]:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise UnsafeUrlError(f"invalid URL: {error}") from error
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("automatic reads require http or https")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeUrlError("credential-bearing URLs are not allowed")
    if not parsed.hostname:
        raise UnsafeUrlError("URL has no hostname")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise UnsafeUrlError(f"local hostname is not allowed: {hostname}")
    port = port or (443 if parsed.scheme == "https" else 80)
    if port not in {80, 443}:
        raise UnsafeUrlError(f"automatic reads allow only ports 80 and 443, not {port}")
    return parsed, hostname, port


async def resolve_host(hostname: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        hostname,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    return list(dict.fromkeys(record[4][0] for record in records))


async def resolve_public_target(url: str, resolver: Resolver = resolve_host) -> PublicTarget:
    parsed, hostname, port = _parse_url(url)
    try:
        ipaddress.ip_address(hostname)
        addresses = [hostname]
    except ValueError:
        addresses = list(await resolver(hostname, port))
    if not addresses:
        raise UnsafeUrlError(f"hostname resolved to no addresses: {hostname}")
    unsafe = [address for address in addresses if not _globally_routable(address)]
    if unsafe:
        raise UnsafeUrlError(f"hostname resolved outside the public internet: {hostname}")

    address = addresses[0]
    address_literal = f"[{address}]" if ":" in address else address
    default_port = 443 if parsed.scheme == "https" else 80
    pinned_netloc = address_literal if port == default_port else f"{address_literal}:{port}"
    logical_host = f"[{hostname}]" if ":" in hostname else hostname
    host_header = logical_host if port == default_port else f"{logical_host}:{port}"
    path = parsed.path or "/"
    pinned_url = urlunsplit((parsed.scheme, pinned_netloc, path, parsed.query, ""))
    logical_url = urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))
    return PublicTarget(logical_url, hostname, port, address, pinned_url, host_header)


async def guarded_get(
    url: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    max_redirects: int = 5,
    max_bytes: int = DEFAULT_MAX_BYTES,
    resolver: Resolver = resolve_host,
    transport: httpx.AsyncBaseTransport | None = None,
) -> GuardedResponse:
    """GET one public URL with pinned DNS and per-hop redirect validation."""

    logical_url = url
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=timeout,
        trust_env=False,
        transport=transport,
    ) as client:
        for redirect_count in range(max_redirects + 1):
            target = await resolve_public_target(logical_url, resolver)
            request_headers = dict(headers or {})
            request_headers["Host"] = target.host_header
            request = client.build_request("GET", target.pinned_url, headers=request_headers)
            request.extensions["sni_hostname"] = target.hostname
            response = await client.send(request, stream=True)
            try:
                if response.status_code in REDIRECT_STATUSES and response.headers.get("location"):
                    if redirect_count >= max_redirects:
                        raise UnsafeUrlError(f"redirect limit exceeded ({max_redirects})")
                    next_url = urljoin(target.logical_url, response.headers["location"])
                    if urlsplit(target.logical_url).scheme == "https" and urlsplit(next_url).scheme == "http":
                        raise UnsafeUrlError("automatic reads do not follow HTTPS-to-HTTP redirects")
                    logical_url = next_url
                    continue

                declared = response.headers.get("content-length")
                if declared and int(declared) > max_bytes:
                    raise UnsafeUrlError(f"response exceeds {max_bytes} bytes")
                body = bytearray()
                if response.is_stream_consumed:
                    body.extend(response.content)
                    if len(body) > max_bytes:
                        raise UnsafeUrlError(f"response exceeds {max_bytes} bytes")
                    decoded = response.text
                else:
                    async for chunk in response.aiter_raw():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise UnsafeUrlError(f"response exceeds {max_bytes} bytes")
                    decoded = httpx.Response(
                        response.status_code,
                        headers=response.headers,
                        content=bytes(body),
                    ).text
                return GuardedResponse(response.status_code, target.logical_url, decoded)
            finally:
                await response.aclose()

    raise UnsafeUrlError("request did not produce a response")
