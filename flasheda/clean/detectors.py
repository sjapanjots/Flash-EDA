"""
flasheda.clean.detectors
~~~~~~~~~~~~~~~~~~~~~~~~
Data-quality checks that run on the sample and decide which cleaning
steps the planner adds. Each returns plain values (no side effects).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

NUMERIC_TEXT_MIN_RATIO = 0.95    # share of values that must parse as numbers
LEADING_ZERO_MAX_RATIO = 0.10    # more than this looks like codes (zip, phone)
RARE_MIN_SHARE = 0.01            # categories below 1% of rows are "rare"
RARE_MIN_UNIQUE = 10             # only group columns with more distinct values
MISSING_MIN_ROWS = 30
MISSING_MIN_GAP = 0.10           # class-share gap between missing / present rows
MISSING_MIN_EFFECT = 0.20        # regression: mean gap in target std units
ID_NAME = re.compile(r"(^|[^a-z])(id|uuid|guid|key)$|_id$|^id_", re.IGNORECASE)


def is_text(s: pd.Series) -> bool:
    values = s.dropna()
    return len(values) > 0 and pd.api.types.infer_dtype(values, skipna=True) == "string"


def looks_like_datetime(s: pd.Series, min_ratio: float = 0.95) -> bool:
    """Text that parses as dates/timestamps (checked on up to 200 values)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not is_text(s):
        return False
    values = s.dropna().astype(str).head(200)
    if not values.str.contains(r"[-/:]").mean() >= min_ratio:
        return False
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= min_ratio)


FREE_TEXT_MIN_WORDS = 3.0
FREE_TEXT_MIN_CHARS = 30.0


def is_free_text(s: pd.Series) -> bool:
    """Text that reads like sentences: ≥ 3 words or ≥ 30 characters on average."""
    if not is_text(s):
        return False
    values = s.dropna().astype(str).head(2_000)
    return bool(values.str.count(r"\S+").mean() >= FREE_TEXT_MIN_WORDS
                or values.str.len().mean() >= FREE_TEXT_MIN_CHARS)


def numeric_text_ratio(s: pd.Series) -> Optional[float]:
    """Share of non-empty text values that parse as numbers, or None if it's not text."""
    if not is_text(s):
        return None
    values = s.dropna().astype(str).str.strip(" ")
    values = values[values != ""]
    if len(values) < MISSING_MIN_ROWS:
        return None
    if values.str.match(r"^-?0\d").mean() > LEADING_ZERO_MAX_RATIO:
        return None
    return float(pd.to_numeric(values, errors="coerce").notna().mean())


def is_numeric_text(s: pd.Series) -> bool:
    ratio = numeric_text_ratio(s)
    return ratio is not None and ratio >= NUMERIC_TEXT_MIN_RATIO


def text_variants(s: pd.Series) -> Optional[Tuple[int, int]]:
    """(distinct before, distinct after) if trimming + lower-casing merges values."""
    if not is_text(s):
        return None
    values = s.dropna().astype(str)
    before = values.nunique()
    after = values.str.strip(" ").str.lower().nunique()
    return (before, after) if after < before else None


def rare_categories(s: pd.Series) -> Optional[Tuple[tuple, str]]:
    """(values to keep, label for the rest) if the column has a long tail."""
    if not is_text(s):
        return None
    values = s.dropna()
    shares = values.value_counts(normalize=True)
    if len(shares) <= RARE_MIN_UNIQUE:
        return None
    keep = shares[shares >= RARE_MIN_SHARE].index   # at most 100 values can each be ≥ 1%
    if len(keep) == 0 or len(keep) == len(shares):
        return None
    other = "other"
    while other in shares.index:
        other = f"_{other}_"
    return tuple(str(v) for v in keep), other


def is_numeric_id(s: pd.Series, name: str) -> bool:
    """Integer-valued, (almost) all unique, and named like an identifier."""
    if not ID_NAME.search(str(name)):
        return False
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    values = s.dropna()
    if len(values) < MISSING_MIN_ROWS or not np.allclose(values, np.round(values)):
        return False
    return values.nunique() / len(values) >= 0.99


def informative_missing(s: pd.Series, y: pd.Series, task: str) -> Optional[float]:
    """
    How differently the target behaves where `s` is missing vs. present,
    or None if the gap is too small to matter.
    """
    mask = y.notna()
    missing = s[mask].isna()
    y = y[mask]
    if missing.sum() < MISSING_MIN_ROWS or (~missing).sum() < MISSING_MIN_ROWS:
        return None

    if task == "regression":
        y = y.astype(float)
        std = y.std()
        if not std:
            return None
        gap = abs(y[missing].mean() - y[~missing].mean()) / std
        return round(float(gap), 3) if gap >= MISSING_MIN_EFFECT else None

    labels = y.astype(str)
    a = labels[missing].value_counts(normalize=True)
    b = labels[~missing].value_counts(normalize=True)
    gap = 0.5 * a.sub(b, fill_value=0).abs().sum()   # total variation distance
    return round(float(gap), 3) if gap >= MISSING_MIN_GAP else None
