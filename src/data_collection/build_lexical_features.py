"""Extract lexical features for every URL in data/processed/dataset.csv.

No network calls, so this runs over the full dataset. Writes
data/processed/lexical_features.csv.
"""
import os

import pandas as pd

from src.features.lexical import extract_features_batch

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


if __name__ == "__main__":
    dataset_path = os.path.join(PROCESSED_DIR, "dataset.csv")
    df = pd.read_csv(dataset_path)
    out = extract_features_batch(df)
    out_path = os.path.join(PROCESSED_DIR, "lexical_features.csv")
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(out)} rows, {out.shape[1]} columns -> {out_path}")
