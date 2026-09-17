"""Extract host-based telemetry for URLs in data/processed/dataset.csv.

Runs concurrently (bounded by CONCURRENCY) with per-host disk caching
(src/utils/cache.py) to stay under WHOIS/DNS rate limits. Writes
data/processed/host_features.csv.

Usage:
    python -m src.data_collection.build_host_features [sample_size]

Defaults to a 200-URL sample — the full ~126k dataset would take hours
and risk WHOIS IP-blocking; raise sample_size once the pipeline is
verified, or pass 0 to run the full dataset.
"""
import asyncio
import os
import sys

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from src.features.host_telemetry import extract_host_features

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
CONCURRENCY = 50


async def _bounded_extract(url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        return await extract_host_features(url, session)


async def run(sample_size: int = 200) -> pd.DataFrame:
    dataset_path = os.path.join(PROCESSED_DIR, "dataset.csv")
    df = pd.read_csv(dataset_path)
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_bounded_extract(url, session, semaphore) for url in df["url"]]
        results = await tqdm_asyncio.gather(*tasks)

    features = pd.DataFrame(results)
    return df.merge(features, on="url", how="left")


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = asyncio.run(run(sample_size=sample))
    out_path = os.path.join(PROCESSED_DIR, "host_features.csv")
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(out)} rows -> {out_path}")
