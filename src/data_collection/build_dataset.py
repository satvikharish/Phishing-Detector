"""Combine raw feeds in data/raw/ into a single labeled dataset.

Output: data/processed/dataset.csv with columns [url, label, source]
label: 1 = phishing, 0 = benign
"""
import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def _latest(prefix: str) -> str:
    matches = sorted(glob.glob(os.path.join(RAW_DIR, f"{prefix}_*")))
    if not matches:
        raise FileNotFoundError(
            f"No files matching {prefix}_*. Run `python -m src.data_collection.fetch_feeds` first."
        )
    return matches[-1]


def load_openphish() -> pd.DataFrame:
    path = _latest("openphish")
    with open(path, "r", errors="ignore") as f:
        urls = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({"url": urls, "label": 1, "source": "openphish"})


def load_phishtank() -> pd.DataFrame:
    path = _latest("phishtank")
    df = pd.read_csv(path, usecols=["url"])
    return pd.DataFrame({"url": df["url"], "label": 1, "source": "phishtank"})


def load_majestic(limit: int = 50_000) -> pd.DataFrame:
    path = _latest("majestic_million")
    df = pd.read_csv(path, usecols=["Domain"], nrows=limit)
    urls = "http://" + df["Domain"].astype(str)
    return pd.DataFrame({"url": urls, "label": 0, "source": "majestic_million"})


def build(limit_benign: int = 50_000) -> pd.DataFrame:
    frames = [load_openphish(), load_phishtank(), load_majestic(limit=limit_benign)]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="url").reset_index(drop=True)
    return combined


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    dataset = build()
    out_path = os.path.join(PROCESSED_DIR, "dataset.csv")
    dataset.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(dataset)} rows -> {out_path}")
    print(dataset["label"].value_counts())
