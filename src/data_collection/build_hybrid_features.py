"""Build the hybrid feature matrix (lexical + host telemetry) for Week 4.

Samples URLs from the dataset (host telemetry requires live network calls,
so this can't run over the full ~126k-row dataset in reasonable time),
extracts lexical features locally and host telemetry over the network, and
merges both into data/processed/hybrid_features.csv.

Usage:
    python -m src.data_collection.build_hybrid_features [sample_size]
"""
import asyncio
import os
import sys

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from src.features.host_telemetry import extract_host_features
from src.features.lexical import extract_features_batch

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
CONCURRENCY = 50


async def _bounded_extract(url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        return await extract_host_features(url, session)


async def _build_host_features(urls: list) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_bounded_extract(url, session, semaphore) for url in urls]
        results = await tqdm_asyncio.gather(*tasks)
    return pd.DataFrame(results)


def run(sample_size: int = 1500) -> pd.DataFrame:
    dataset_path = os.path.join(PROCESSED_DIR, "dataset.csv")
    df = pd.read_csv(dataset_path)
    df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    lexical = extract_features_batch(df)
    host = asyncio.run(_build_host_features(df["url"].tolist()))

    return lexical.merge(host, on="url", how="left")


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    out = run(sample_size=sample)
    out_path = os.path.join(PROCESSED_DIR, "hybrid_features.csv")
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(out)} rows, {out.shape[1]} columns -> {out_path}")
