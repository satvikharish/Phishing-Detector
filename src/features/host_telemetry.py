"""Host-based telemetry feature extraction (active network lookups).

DNS record flags, WHOIS domain age, SSL/TLS validity, and HTTP response
headers, per the proposal's "Host-Based Telemetry & Dynamic Footprinting"
section. Every lookup is cached to disk (src/utils/cache.py) since
WHOIS/DNS endpoints rate-limit aggressively, and every lookup fails soft
(returns default values) rather than raising, since network telemetry is
expected to be flaky at this scale.
"""
import asyncio
import socket
import ssl
from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp
import dns.asyncresolver
import dns.exception
import whois

from src.utils.cache import cache_get, cache_set

DNS_TIMEOUT = 3.0
SSL_TIMEOUT = 3.0
HTTP_TIMEOUT = 5.0
WHOIS_TIMEOUT = 7.0


def _hostname_of(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return parsed.hostname or url


async def get_dns_flags(hostname: str) -> dict:
    cached = cache_get("dns", hostname)
    if cached is not None:
        return cached

    resolver = dns.asyncresolver.Resolver()
    resolver.timeout = DNS_TIMEOUT
    resolver.lifetime = DNS_TIMEOUT

    result = {}
    for record_type, key in (("A", "has_a_record"), ("MX", "has_mx_record"), ("TXT", "has_txt_record")):
        try:
            await resolver.resolve(hostname, record_type)
            result[key] = True
        except (dns.exception.DNSException, asyncio.TimeoutError, OSError):
            result[key] = False

    cache_set("dns", hostname, result)
    return result


def _whois_lookup_sync(hostname: str) -> dict:
    try:
        w = whois.whois(hostname)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0] if created else None
        if created is None:
            return {"domain_age_days": None, "has_whois_record": False}
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - created).days
        return {"domain_age_days": age_days, "has_whois_record": True}
    except Exception:
        return {"domain_age_days": None, "has_whois_record": False}


async def get_whois_age(hostname: str) -> dict:
    cached = cache_get("whois", hostname)
    if cached is not None:
        return cached
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_whois_lookup_sync, hostname), timeout=WHOIS_TIMEOUT
        )
    except asyncio.TimeoutError:
        result = {"domain_age_days": None, "has_whois_record": False}
    cache_set("whois", hostname, result)
    return result


def _ssl_lookup_sync(hostname: str) -> dict:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=SSL_TIMEOUT) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
        not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        days_left = (not_after - datetime.now(timezone.utc)).days
        return {"has_valid_ssl": days_left > 0, "ssl_days_until_expiry": days_left}
    except Exception:
        return {"has_valid_ssl": False, "ssl_days_until_expiry": None}


async def get_ssl_info(hostname: str) -> dict:
    cached = cache_get("ssl", hostname)
    if cached is not None:
        return cached
    result = await asyncio.to_thread(_ssl_lookup_sync, hostname)
    cache_set("ssl", hostname, result)
    return result


async def get_http_headers(url: str, session: aiohttp.ClientSession) -> dict:
    hostname = _hostname_of(url)
    cached = cache_get("headers", hostname)
    if cached is not None:
        return cached

    result = {
        "http_status": None,
        "server_header_present": False,
        "has_hsts": False,
        "has_csp": False,
        "has_x_frame_options": False,
    }
    try:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        async with session.get(url, timeout=timeout, allow_redirects=True, ssl=False) as resp:
            headers = resp.headers
            result = {
                "http_status": resp.status,
                "server_header_present": "Server" in headers,
                "has_hsts": "Strict-Transport-Security" in headers,
                "has_csp": "Content-Security-Policy" in headers,
                "has_x_frame_options": "X-Frame-Options" in headers,
            }
    except Exception:
        pass

    cache_set("headers", hostname, result)
    return result


async def extract_host_features(url: str, session: aiohttp.ClientSession) -> dict:
    """Combine all host-based telemetry for a single URL."""
    hostname = _hostname_of(url)
    dns_res, whois_res, ssl_res, headers_res = await asyncio.gather(
        get_dns_flags(hostname),
        get_whois_age(hostname),
        get_ssl_info(hostname),
        get_http_headers(url, session),
    )
    combined = {"url": url}
    combined.update(dns_res)
    combined.update(whois_res)
    combined.update(ssl_res)
    combined.update(headers_res)
    return combined
