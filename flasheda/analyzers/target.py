"""
flasheda.analyzers.target
~~~~~~~~~~~~~~~~~~~~~~~~~
Analysis of the column you want to predict:

  - task type        binary / multiclass classification or regression
  - balance          class shares and imbalance ratio
  - predictive power how well each feature alone predicts the target,
                     scored 0..1 on a held-out part of the sample
  - leakage          features that predict the target almost perfectly,
                     usually because they are recorded after the outcome
  - recommendations  evaluation metric and validation split

Predictive power is numpy/pandas only: each feature is bucketed (quantile
bins for numbers, values for categories), a lookup table is learned on 70%
of the sample and scored on the other 30%. Unseen buckets fall back to the
prior, so ID-like columns score ~0 instead of looking perfect.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..clean.detectors import looks_like_datetime

LEAK_THRESHOLD = 0.8         # power at or above this → possible leakage
MAX_CLASSES = 20             # numeric targets with more values → regression
NUMERIC_BINS = 20
SMOOTHING = 10.0             # pseudo-count pulling small buckets toward the prior
MIN_ROWS = 30
TIME_HINTS = ("date", "time", "timestamp", "_at", "_on")


def analyze(df: pd.DataFrame, target: str, random_state: int = 42) -> Dict[str, Any]:
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not found. Columns: {list(df.columns)}")

    y_all = df[target]
    null_count = int(y_all.isna().sum())
    y = y_all.dropna()
    task = detect_task(y)

    result: Dict[str, Any] = {
        "target": target,
        "task": task,
        "count": int(len(y)),
        "null_count": null_count,
        "null_pct": round(null_count / len(df) * 100, 2) if len(df) else 0.0,
    }

    if task == "regression":
        result["target_stats"] = _regression_stats(y)
        result["imbalanced"] = False
    else:
        result.update(_class_balance(y))

    features = [c for c in df.columns if c != target]
    rng = np.random.default_rng(random_state)
    power = []
    for col in features:
        score = predictive_power(df[col], y_all, task, rng)
        if score is not None:
            power.append({"column": col, "score": round(score, 4)})
    power.sort(key=lambda p: p["score"], reverse=True)

    result["feature_power"] = power
    result["leakage_columns"] = [
        {"column": p["column"], "score": p["score"],
         "reason": f"predicts the target almost perfectly on its own (power {p['score']})"}
        for p in power if p["score"] >= LEAK_THRESHOLD
    ]
    result["time_columns"] = find_time_columns(df[features])
    result["recommended_metric"] = recommend_metric(result)
    result["recommended_split"] = recommend_split(result)
    return result


def detect_task(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "binary" if y.nunique() <= 2 else "multiclass"
    if y.nunique() <= 2:
        return "binary"
    integer_like = np.allclose(y, np.round(y))
    if integer_like and y.nunique() <= MAX_CLASSES:
        return "multiclass"
    return "regression"


def predictive_power(
    x: pd.Series, y: pd.Series, task: str, rng: np.random.Generator
) -> Optional[float]:
    """
    Single-feature predictive power in [0, 1], measured on a held-out split.

    classification: Brier skill score of smoothed per-bucket class
                    probabilities vs. the class prior
    regression:     held-out R² of per-bucket means (and of a linear fit
                    for numeric features), whichever is higher
    """
    mask = y.notna().to_numpy()
    x, y = x[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    n = len(y)
    if n < MIN_ROWS:
        return None

    key = _bucket(x).to_numpy()
    order = rng.permutation(n)
    cut = int(n * 0.7)
    train, test = order[:cut], order[cut:]

    if task == "regression":
        yv = y.astype(float).to_numpy()
        y_train, y_test = yv[train], yv[test]
        sst = float(((y_test - y_train.mean()) ** 2).sum())
        if sst == 0:
            return 0.0
        means = pd.Series(y_train).groupby(key[train]).mean()
        pred = pd.Series(key[test]).map(means).fillna(y_train.mean()).to_numpy()
        best = 1 - float(((y_test - pred) ** 2).sum()) / sst
        if _is_number(x):
            xv = x.astype(float).to_numpy()
            ok_train = ~np.isnan(xv[train])
            if ok_train.sum() >= MIN_ROWS and np.ptp(xv[train][ok_train]) > 0:
                slope, icpt = np.polyfit(xv[train][ok_train], y_train[ok_train], 1)
                lin = np.where(np.isnan(xv[test]), y_train.mean(), slope * xv[test] + icpt)
                best = max(best, 1 - float(((y_test - lin) ** 2).sum()) / sst)
        return float(np.clip(best, 0, 1))

    labels = y.astype(str).to_numpy()
    classes, codes = np.unique(labels[train], return_inverse=True)
    onehot_train = np.eye(len(classes))[codes]
    prior = onehot_train.mean(axis=0)

    sums = pd.DataFrame(onehot_train).groupby(key[train]).sum()
    counts = sums.sum(axis=1).to_numpy()[:, None]
    probs = (sums.to_numpy() + SMOOTHING * prior) / (counts + SMOOTHING)
    table = pd.DataFrame(probs, index=sums.index)

    pred = table.reindex(key[test]).to_numpy(copy=True)
    unseen = np.isnan(pred[:, 0])
    pred[unseen] = prior
    truth = (labels[test][:, None] == classes[None, :]).astype(float)

    brier = float(((pred - truth) ** 2).sum(axis=1).mean())
    brier_base = float(((prior - truth) ** 2).sum(axis=1).mean())
    if brier_base == 0:
        return 0.0
    return float(np.clip(1 - brier / brier_base, 0, 1))


def _is_number(x: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)


def _bucket(x: pd.Series) -> pd.Series:
    """Discrete keys for a feature; nulls get their own bucket."""
    if _is_number(x) and x.nunique() > NUMERIC_BINS:
        codes = pd.qcut(x, q=NUMERIC_BINS, labels=False, duplicates="drop")
        return codes.astype("float64").fillna(-1).astype(int).astype(str)
    return x.astype(str).where(x.notna(), "<null>")


def _class_balance(y: pd.Series) -> Dict[str, Any]:
    counts = y.astype(str).value_counts()
    n = int(counts.sum())
    ratio = float(counts.iloc[0] / counts.iloc[-1]) if len(counts) > 1 else 1.0
    minority_pct = round(counts.iloc[-1] / n * 100, 2)
    return {
        "classes": [
            {"value": v, "count": int(c), "pct": round(c / n * 100, 2)}
            for v, c in counts.head(MAX_CLASSES).items()
        ],
        "n_classes": int(len(counts)),
        "imbalance_ratio": round(ratio, 2),
        "minority_pct": minority_pct,
        "imbalanced": ratio > 4,
    }


def _regression_stats(y: pd.Series) -> Dict[str, float]:
    y = y.astype(float)
    return {
        "mean": round(float(y.mean()), 4),
        "std": round(float(y.std()), 4),
        "min": round(float(y.min()), 4),
        "max": round(float(y.max()), 4),
        "skewness": round(float(y.skew()), 4),
    }


def find_time_columns(df: pd.DataFrame) -> List[str]:
    found = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(col)
            continue
        name = str(col).lower()
        if any(h in name for h in TIME_HINTS) and looks_like_datetime(s):
            found.append(col)
    return found


def recommend_metric(r: Dict[str, Any]) -> str:
    task = r["task"]
    if task == "regression":
        stats = r["target_stats"]
        if stats["skewness"] > 1 and stats["min"] >= 0:
            return "MAE or RMSLE (target is right-skewed, so RMSE would be dominated by large values)"
        return "RMSE (and MAE for an outlier-robust view)"
    if task == "binary":
        if r["imbalanced"]:
            return (f"PR-AUC / average precision (minority class is {r['minority_pct']}%, "
                    "so accuracy and ROC-AUC look optimistic)")
        return "ROC-AUC (classes are balanced)"
    if r["imbalanced"]:
        return "macro-F1 (classes are imbalanced, so accuracy hides small classes)"
    return "accuracy, plus macro-F1"


def recommend_split(r: Dict[str, Any]) -> str:
    if r["time_columns"]:
        cols = ", ".join(map(str, r["time_columns"]))
        return f"time-based split on {cols}: train on the past, test on the future"
    if r["task"] == "regression":
        return "K-fold cross-validation (5 folds)"
    return "stratified K-fold cross-validation (5 folds), so every fold keeps the class balance"
