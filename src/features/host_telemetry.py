"""Host-based telemetry feature extraction (active network lookups).

Week 2 target per proposal timeline: DNS record flags, WHOIS domain age,
ASN, SSL/TLS validity, HTTP headers. Implement using asyncio/aiohttp with
local caching (see src/utils/cache.py) to avoid rate-limiting.
"""
import asyncio

import aiohttp


async def get_dns_flags(hostname: str) -> dict:
    """Return presence of A / MX / TXT records for `hostname`.

    TODO (Week 2): implement with dnspython's asyncio resolver.
    """
    raise NotImplementedError


async def get_whois_age(hostname: str) -> dict:
    """Return domain creation/expiration age in days for `hostname`.

    TODO (Week 2): implement with python-whois, cache results locally
    since WHOIS servers rate-limit aggressively.
    """
    raise NotImplementedError


async def get_ssl_info(hostname: str) -> dict:
    """Return SSL/TLS certificate validity info for `hostname`.

    TODO (Week 2): open a TLS socket (ssl.create_default_context) and
    inspect the certificate's issuer/expiry.
    """
    raise NotImplementedError


async def get_http_headers(url: str, session: aiohttp.ClientSession) -> dict:
    """Return relevant HTTP response headers for `url`.

    TODO (Week 2): async GET with timeout + retry, extract headers like
    Server, X-Frame-Options, Content-Security-Policy.
    """
    raise NotImplementedError


async def extract_host_features(url: str) -> dict:
    """Combine all host-based telemetry for a single URL."""
    raise NotImplementedError
