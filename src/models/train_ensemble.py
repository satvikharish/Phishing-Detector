"""Week 4-5: weighted XGBoost + Random Forest ensemble on the hybrid
(lexical + host telemetry) feature set, with hyperparameter tuning and
10-fold CV benchmarking against single-modality baselines.
"""
import os
import sys
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from src.models.evaluate import cross_validate, tune_hyperparameters
from src.models.features import to_xy

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

LEXICAL_COLUMNS = [
    "url_length", "hostname_length", "path_length", "query_length", "num_dots",
    "num_hyphens", "num_underscores", "num_slashes", "num_digits",
    "num_special_chars", "digit_ratio", "shannon_entropy", "num_subdomains",
    "has_ip_address", "has_https", "suspicious_keyword_count",
]

XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


def build_tuned_ensemble(X, y) -> VotingClassifier:
    base_xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    tuned_xgb, best_params = tune_hyperparameters(base_xgb, XGB_PARAM_DISTRIBUTIONS, X, y, n_iter=20)
    print(f"[xgboost] best params (tuned for precision): {best_params}")

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)

    # Soft-vote weighted 2:1 toward XGBoost, since it's tuned specifically
    # for this feature set; Random Forest still contributes to reduce variance.
    return VotingClassifier(
        estimators=[("xgboost", tuned_xgb), ("random_forest", rf)],
        voting="soft",
        weights=[2, 1],
    )


def _cv_summary(model, X, y, name: str, use_smote: bool) -> pd.DataFrame:
    cv = cross_validate(model, X, y, n_splits=10, use_smote=use_smote)
    summary = cv[cv["fold"] == "mean"].copy()
    summary.insert(0, "model", name)
    print(f"[{name}] " + ", ".join(f"{k}={v:.3f}" for k, v in summary.drop(columns=["fold", "model"]).iloc[0].items()))
    return summary


def run(sample_size: int = None, use_smote: bool = False) -> pd.DataFrame:
    features_path = os.path.join(PROCESSED_DIR, "hybrid_features.csv")
    df = pd.read_csv(features_path)
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    X, y, feature_cols = to_xy(df)
    host_cols = [c for c in feature_cols if c not in LEXICAL_COLUMNS]

    results = []

    # Single-modality baselines on the same rows, for a fair benchmark.
    results.append(_cv_summary(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        X[LEXICAL_COLUMNS], y, "random_forest_lexical_only", use_smote,
    ))
    results.append(_cv_summary(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        X[host_cols], y, "random_forest_host_only", use_smote,
    ))

    # Hybrid ensemble (Week 4/5 deliverable).
    ensemble = build_tuned_ensemble(X, y)
    results.append(_cv_summary(ensemble, X, y, "xgboost_rf_ensemble_hybrid", use_smote))

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    results = run(sample_size=sample)
    out_path = os.path.join(PROCESSED_DIR, "ensemble_results.csv")
    results.to_csv(out_path, index=False)
    print(f"[ok] wrote ensemble results -> {out_path}")
