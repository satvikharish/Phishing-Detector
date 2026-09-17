"""Disk-based JSON cache for host telemetry lookups.

Keyed by (namespace, key), e.g. ("whois", "example.com"). Avoids
re-querying rate-limited WHOIS/DNS/SSL/HTTP endpoints across runs.
"""
import json
import os
import time

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 1 week


def _path(namespace: str, key: str) -> str:
    safe_key = key.replace("/", "_")
    ns_dir = os.path.join(CACHE_DIR, namespace)
    os.makedirs(ns_dir, exist_ok=True)
    return os.path.join(ns_dir, f"{safe_key}.json")


def cache_get(namespace: str, key: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
    path = _path(namespace, key)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > ttl_seconds:
        return None
    with open(path, "r") as f:
        return json.load(f)


def cache_set(namespace: str, key: str, value: dict) -> None:
    path = _path(namespace, key)
    with open(path, "w") as f:
        json.dump(value, f)
