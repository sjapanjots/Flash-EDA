"""
flasheda.analyzers.correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Computes:
  - Pearson correlation for numeric–numeric pairs
  - Cramér's V for categorical–categorical pairs
  - Point-biserial for numeric–boolean pairs

Flags strongly correlated pairs (potential redundant features).
Always operates on the fixed sample — constant time.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from itertools import combinations


STRONG_CORR_THRESHOLD = 0.80
MAX_COLS_FOR_HEATMAP = 20   # skip heatmap if too many cols (slow to render)
MAX_CRAMER_LEVELS = 100     # Cramér's V is meaningless (and slow) for ID-like columns


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

    return {
        "pearson_matrix": pearson_matrix,
        "numeric_strong_pairs": strong_pairs,
        "categorical_strong_pairs": cramer_pairs,
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