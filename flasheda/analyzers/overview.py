"""
flasheda.analyzers.overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fast dataset-level summary: shape, dtypes, memory, duplicates.
Operates on the fixed-size sample — O(n_sample) always.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def analyze(df: pd.DataFrame, original_shape: tuple) -> Dict[str, Any]:
    """
    Parameters
    ----------
    df             : sampled DataFrame
    original_shape : (rows, cols) of the original dataset before sampling
    """
    original_rows, ncols = original_shape
    sample_rows = len(df)

    dtype_counts: Dict[str, int] = {}
    for dtype in df.dtypes:
        key = _simplify_dtype(dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1

    mem_bytes = df.memory_usage(deep=True).sum()
    # Extrapolate memory to full dataset
    scale = original_rows / sample_rows if sample_rows > 0 else 1
    est_full_mem = mem_bytes * scale

    dup_count = df.duplicated().sum()
    dup_pct = round(dup_count / sample_rows * 100, 2) if sample_rows > 0 else 0.0

    col_types = {col: _simplify_dtype(df[col].dtype) for col in df.columns}

    return {
        "original_rows": original_rows,
        "sample_rows": sample_rows,
        "columns": ncols,
        "dtype_counts": dtype_counts,
        "col_types": col_types,
        "memory_sample_bytes": int(mem_bytes),
        "memory_estimated_full_bytes": int(est_full_mem),
        "duplicate_rows_in_sample": int(dup_count),
        "duplicate_pct_in_sample": dup_pct,
        "column_names": list(df.columns),
    }


def _simplify_dtype(dtype) -> str:
    s = str(dtype)
    if "int" in s:
        return "integer"
    if "float" in s:
        return "float"
    if "bool" in s:
        return "boolean"
    if "datetime" in s:
        return "datetime"
    if "object" in s or "string" in s or s == "str":  # pandas 3 string dtype prints as "str"
        return "string/object"
    if "category" in s:
        return "category"
    return s