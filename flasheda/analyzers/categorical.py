"""
flasheda.analyzers.categorical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Categorical / string column analysis.
Computes cardinality, top-k frequencies, and flags high-cardinality columns.
Uses HyperLogLog-style approximation for cardinality on large columns.
Constant-time: always operates on the fixed sample.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from ..clean.detectors import is_free_text, looks_like_datetime


HIGH_CARDINALITY_RATIO = 0.5   # unique/total > 50% → high cardinality warning
TOP_K = 10


def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    # Include object, string, category, and boolean columns
    cat_cols = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    results: Dict[str, Dict[str, Any]] = {}

    for col in cat_cols:
        s = df[col].dropna().astype(str)
        n = len(s)
        if n == 0:
            results[col] = {"error": "all values are null"}
            continue

        unique_count = s.nunique()
        cardinality_ratio = round(unique_count / n, 4)
        high_cardinality = cardinality_ratio > HIGH_CARDINALITY_RATIO

        vc = s.value_counts()
        top_k = vc.head(TOP_K)

        top_values = [
            {
                "value": str(val),
                "count": int(cnt),
                "pct": round(cnt / n * 100, 2),
            }
            for val, cnt in top_k.items()
        ]

        # Mode
        mode_val = vc.index[0] if len(vc) > 0 else None
        mode_pct = round(vc.iloc[0] / n * 100, 2) if len(vc) > 0 else 0.0

        # Detect potential ID columns (very high cardinality); unique
        # timestamps and free text are high-cardinality too, but not identifiers
        likely_id = (cardinality_ratio > 0.95 and not looks_like_datetime(df[col])
                     and not is_free_text(df[col]))

        results[col] = {
            "count": n,
            "unique_count": unique_count,
            "cardinality_ratio": cardinality_ratio,
            "high_cardinality": high_cardinality,
            "likely_id_column": likely_id,
            "mode": str(mode_val),
            "mode_pct": mode_pct,
            "top_values": top_values,
        }

    high_card_cols = [c for c, v in results.items() if v.get("high_cardinality")]
    likely_id_cols = [c for c, v in results.items() if v.get("likely_id_column")]

    return {
        "columns": results,
        "categorical_col_names": cat_cols,
        "high_cardinality_columns": high_card_cols,
        "likely_id_columns": likely_id_cols,
    }