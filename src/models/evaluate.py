"""Shared evaluation utilities: 10-fold CV, class-imbalance handling, and
hyperparameter tuning (Week 5 of the proposal timeline).
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def _false_positive_rate(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def cross_validate(model, X, y, n_splits: int = 10, use_smote: bool = False, random_state: int = 42) -> pd.DataFrame:
    """Stratified k-fold CV, returning per-fold metrics plus a `mean` row.

    Precision/recall/F1 are reported for the phishing (positive) class,
    since minimizing false positives on benign sites is the proposal's
    stated real-world priority.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_smote:
            X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        y_proba = fold_model.predict_proba(X_test)[:, 1] if hasattr(fold_model, "predict_proba") else y_pred

        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auroc": roc_auc_score(y_test, y_proba),
            "false_positive_rate": _false_positive_rate(y_test, y_pred),
        })

    results = pd.DataFrame(rows)
    mean_row = results.drop(columns="fold").mean().to_dict()
    mean_row["fold"] = "mean"
    results = pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)
    return results


def tune_hyperparameters(model, param_distributions: dict, X, y, n_iter: int = 20, random_state: int = 42):
    """Randomized search over `param_distributions`, optimizing precision
    (the proposal's stated priority: minimize false positives).
    """
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
