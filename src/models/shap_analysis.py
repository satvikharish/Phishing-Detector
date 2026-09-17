"""Week 6: SHAP feature-importance analysis for the tuned XGBoost model.

Explains the XGBoost component of the ensemble directly (shap.TreeExplainer
is fast and exact for tree models; explaining the full soft-voting ensemble
would require the much slower model-agnostic KernelExplainer for little
extra insight, since XGBoost is the dominant, 2x-weighted vote).
"""
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.models.evaluate import tune_hyperparameters
from src.models.features import to_xy
from src.models.train_ensemble import XGB_PARAM_DISTRIBUTIONS

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")


def run(sample_size: int = None):
    from xgboost import XGBClassifier

    features_path = os.path.join(PROCESSED_DIR, "hybrid_features.csv")
    df = pd.read_csv(features_path)
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    X, y, feature_cols = to_xy(df)

    base_xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    tuned_xgb, best_params = tune_hyperparameters(base_xgb, XGB_PARAM_DISTRIBUTIONS, X, y, n_iter=20)
    tuned_xgb.fit(X, y)

    explainer = shap.TreeExplainer(tuned_xgb)
    shap_values = explainer.shap_values(X)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    summary_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.savefig(summary_path, bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_cols, plot_type="bar", show=False)
    bar_path = os.path.join(FIGURES_DIR, "shap_importance_bar.png")
    plt.savefig(bar_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[ok] wrote {summary_path}")
    print(f"[ok] wrote {bar_path}")
    return shap_values, feature_cols


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    run()
