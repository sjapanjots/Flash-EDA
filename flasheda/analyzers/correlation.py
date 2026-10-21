"""
flasheda.analyzers.correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Computes:
  - Pearson correlation for numeric–numeric pairs
  - Cramér's V for categorical–categorical pairs
  - Point-biserial for numeric–boolean pairs (Pearson with the boolean as 0/1)

Flags strongly correlated pairs (potential redundant features).
Always operates on the fixed sample — constant time.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from itertools import combinations


STRONG_CORR_THRESHOLD = 0.80
MAX_COLS_FOR_HEATMAP = 20   # skip heatmap if too many cols (slow to render)
MAX_CRAMER_LEVELS = 100     # Cramér's V is meaningless (and slow) for ID-like columns
MAX_BOOL_COLS = 10


def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cramer_cols = [c for c in cat_cols if 2 <= df[c].nunique() <= MAX_CRAMER_LEVELS]

    # ── Numeric correlations ────────────────────────────────────────────────
    pearson_matrix: Dict[str, Any] = {}
    strong_pairs: List[Dict[str, Any]] = []

    if len(num_cols) >= 2:
        num_df = df[num_cols].copy()
        corr = num_df.corr(method="pearson")
        pearson_matrix = {
            col: {
                other: round(float(corr.loc[col, other]), 4)
                for other in num_cols
            }
            for col in num_cols
        }

        for a, b in combinations(num_cols, 2):
            v = abs(float(corr.loc[a, b]))
            if v >= STRONG_CORR_THRESHOLD and not np.isnan(v):
                strong_pairs.append({"col_a": a, "col_b": b, "pearson_r": round(float(corr.loc[a, b]), 4)})

    # ── Cramér's V for categoricals ─────────────────────────────────────────
    cramer_pairs: List[Dict[str, Any]] = []
    if len(cramer_cols) >= 2:
        for a, b in combinations(cramer_cols[:10], 2):   # cap at 10 cols
            v = _cramers_v(df[a], df[b])
            if v is not None and v >= STRONG_CORR_THRESHOLD:
                cramer_pairs.append({"col_a": a, "col_b": b, "cramers_v": round(v, 4)})

    # ── Heatmap data (numeric only, capped) ─────────────────────────────────
    heatmap_cols = num_cols[:MAX_COLS_FOR_HEATMAP]
    heatmap_data: Dict[str, Any] = {}
    if len(heatmap_cols) >= 2:
        hm_corr = df[heatmap_cols].corr(method="pearson")
        heatmap_data = {
            "columns": heatmap_cols,
            "matrix": [[round(float(hm_corr.loc[r, c]), 3) for c in heatmap_cols] for r in heatmap_cols],
        }

    # ── Point-biserial for numeric–boolean pairs ────────────────────────────
    biserial_pairs: List[Dict[str, Any]] = []
    for b in [c for c in df.columns if _is_boolean(df[c])][:MAX_BOOL_COLS]:
        flag = _as_01(df[b])
        for a in num_cols[:MAX_COLS_FOR_HEATMAP]:
            r = _point_biserial(df[a], flag)
            if r is not None and abs(r) >= STRONG_CORR_THRESHOLD:
                biserial_pairs.append({"numeric": a, "boolean": b, "point_biserial_r": round(r, 4)})

    return {
        "pearson_matrix": pearson_matrix,
        "numeric_strong_pairs": strong_pairs,
        "categorical_strong_pairs": cramer_pairs,
        "numeric_boolean_strong_pairs": biserial_pairs,
        "heatmap": heatmap_data,
        "numeric_col_names": num_cols,
    }


def _cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Cramér's V: symmetric measure of association for categorical columns."""
    try:
        from scipy.stats import chi2_contingency
        ct = pd.crosstab(a.astype(str), b.astype(str))
        chi2, _, _, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        phi2 = chi2 / n
        r, k = ct.shape
        phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r_corr = r - (r - 1) ** 2 / (n - 1)
        k_corr = k - (k - 1) ** 2 / (n - 1)
        denom = min(k_corr - 1, r_corr - 1)
        if denom <= 0:
            return None
        return float(np.sqrt(phi2_corr / denom))
    except Exception:
        return None

def _is_boolean(s: pd.Series) -> bool:
    """bool dtype, or text holding only true/false."""
    if pd.api.types.is_bool_dtype(s):
        return True
    values = s.dropna()
    if not len(values) or pd.api.types.is_numeric_dtype(values):
        return False
    return bool(values.astype(str).str.strip().str.lower().isin(["true", "false"]).all())


def _as_01(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    text = s.astype("string").str.strip().str.lower()
    return text.map({"true": 1.0, "false": 0.0}).astype(float)


def _point_biserial(x: pd.Series, flag: pd.Series) -> float | None:
    """Point-biserial r: Pearson between a numeric column and a 0/1 flag."""
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "b": flag}).dropna()
    if len(pair) < 3 or pair["x"].std() == 0 or pair["b"].nunique() < 2:
        return None
    from scipy.stats import pointbiserialr
    return float(pointbiserialr(pair["b"], pair["x"]).statistic)
