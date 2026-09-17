"""Lexical URL feature extraction (static, no network calls).

Covers the "Static Lexical Feature Analysis" features from the outline:
character distribution, Shannon entropy, URL length, suspicious keyword
vectors, subdomain structure.
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import tldextract

SUSPICIOUS_KEYWORDS = [
    "login", "secure", "account", "update", "verify", "banking", "confirm",
    "signin", "password", "billing", "webscr", "ebayisapi", "suspend",
    "wallet", "alert", "invoice",
]

IP_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    ext = tldextract.extract(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9.\-/:_]", url)),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "shannon_entropy": shannon_entropy(url),
        "num_subdomains": len(ext.subdomain.split(".")) if ext.subdomain else 0,
        "has_ip_address": bool(IP_PATTERN.match(hostname)),
        "has_https": parsed.scheme == "https",
        "suspicious_keyword_count": sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()),
    }


def extract_features_batch(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    df = df.reset_index(drop=True)
    features = df[url_col].apply(extract_features).apply(pd.Series)
    return pd.concat([df, features], axis=1)


if __name__ == "__main__":
    sample = "http://secure-login.paypal-verify.com/update/account?id=1234"
    print(extract_features(sample))
