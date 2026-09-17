"""Turn a features DataFrame (lexical and/or host telemetry columns) into
a numeric (X, y) matrix ready for sklearn/XGBoost.
"""
import pandas as pd

NON_FEATURE_COLUMNS = {"url", "label", "source"}


def to_xy(df: pd.DataFrame):
    """Split a features DataFrame into (X, y, feature_names).

    Booleans are cast to int; missing values (e.g. failed WHOIS/SSL
    lookups) are filled with the column median so rows aren't dropped.
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].copy()

    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    y = df["label"].astype(int)
    return X, y, feature_cols
