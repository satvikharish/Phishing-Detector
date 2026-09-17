"""Week 6: feature-extraction latency trade-offs (lexical vs. host telemetry).

Measures wall-clock time per URL for each feature source, to quantify the
deployment cost of adding host-based telemetry (the proposal's stated
"extraction latency trade-offs" discussion point).
"""
import asyncio
import os
import time

import aiohttp
import matplotlib.pyplot as plt
import pandas as pd

from src.features.host_telemetry import extract_host_features
from src.features.lexical import extract_features

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")


def measure_lexical_latency(urls: list) -> list:
    latencies = []
    for url in urls:
        start = time.perf_counter()
        extract_features(url)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    return latencies


async def _measure_host_latency_async(urls: list) -> list:
    latencies = []
    connector = aiohttp.TCPConnector(limit=50, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for url in urls:
            start = time.perf_counter()
            await extract_host_features(url, session)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
    return latencies


def measure_host_latency(urls: list) -> list:
    return asyncio.run(_measure_host_latency_async(urls))


def run(sample_size: int = 50) -> pd.DataFrame:
    dataset_path = os.path.join(PROCESSED_DIR, "dataset.csv")
    df = pd.read_csv(dataset_path).sample(sample_size, random_state=7)
    urls = df["url"].tolist()

    lexical_ms = measure_lexical_latency(urls)
    host_ms = measure_host_latency(urls)

    results = pd.DataFrame({
        "lexical_ms": lexical_ms,
        "host_telemetry_ms": host_ms,
    })

    os.makedirs(FIGURES_DIR, exist_ok=True)
    means = results.mean()
    plt.figure()
    plt.bar(["Lexical\n(no network)", "Host telemetry\n(DNS/WHOIS/SSL/HTTP)"], means.values)
    plt.yscale("log")
    plt.ylabel("Mean extraction latency per URL, log scale (ms)")
    plt.title("Feature extraction latency: lexical vs. host telemetry")
    for i, v in enumerate(means.values):
        plt.text(i, v, f"{v:.1f} ms", ha="center", va="bottom")
    fig_path = os.path.join(FIGURES_DIR, "latency_comparison.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(results.describe())
    print(f"[ok] wrote {fig_path}")
    return results


if __name__ == "__main__":
    import sys
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out = run(sample_size=sample)
    out_path = os.path.join(PROCESSED_DIR, "latency_results.csv")
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")
