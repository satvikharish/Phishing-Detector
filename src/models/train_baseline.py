"""Week 3: baseline models (Decision Tree, Logistic Regression, Random
Forest) trained on lexical features alone, evaluated with 10-fold CV.
"""
import os
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import cross_validate
from src.models.features import to_xy

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

BASELINE_MODELS = {
    "decision_tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
}


def run(sample_size: int = None) -> pd.DataFrame:
    features_path = os.path.join(PROCESSED_DIR, "lexical_features.csv")
    df = pd.read_csv(features_path)
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    X, y, _ = to_xy(df)

    all_results = []
    for name, model in BASELINE_MODELS.items():
        if isinstance(model, LogisticRegression):
            X_input = StandardScaler().fit_transform(X)
        else:
            X_input = X
        cv_results = cross_validate(model, X_input, y, n_splits=10)
        summary = cv_results[cv_results["fold"] == "mean"].copy()
        summary.insert(0, "model", name)
        all_results.append(summary)
        print(f"[{name}] " + ", ".join(f"{k}={v:.3f}" for k, v in summary.drop(columns=["fold", "model"]).iloc[0].items()))

    return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    results = run()
    out_path = os.path.join(PROCESSED_DIR, "baseline_results.csv")
    results.to_csv(out_path, index=False)
    print(f"[ok] wrote baseline results -> {out_path}")
