"""Download raw benign/phishing URL feeds into data/raw/.

Sources (per proposal):
- OpenPhish free feed (phishing, live)
- PhishTank verified feed (phishing, live)
- Majestic Million (benign, top-ranked domains)
"""
import os
from datetime import date

import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

FEEDS = {
    "openphish.txt": "https://openphish.com/feed.txt",
    "phishtank.csv": "http://data.phishtank.com/data/online-valid.csv",
    "majestic_million.csv": "https://downloads.majestic.com/majestic_million.csv",
}


def _dated_path(filename: str) -> str:
    stem, ext = os.path.splitext(filename)
    return os.path.join(RAW_DIR, f"{stem}_{date.today().isoformat()}{ext}")


def fetch(filename: str, url: str, force: bool = False) -> str:
    """Download `url` to data/raw/, skipping if today's copy already exists."""
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = _dated_path(filename)
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} already downloaded today")
        return out_path

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"[ok] {url} -> {out_path} ({len(resp.content)} bytes)")
    return out_path


def fetch_all(force: bool = False) -> dict:
    return {name: fetch(name, url, force=force) for name, url in FEEDS.items()}


if __name__ == "__main__":
    fetch_all()
