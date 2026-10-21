"""
flasheda.features
~~~~~~~~~~~~~~~~~
Plans model-ready features on top of the cleaning plan:

  dates       ISO date/timestamp text (or datetime columns) → year, month,
              day, day of week (and hour when times vary); original dropped
  text        free-text columns → length and word count; columns with
              personal data (emails, phones, …) are dropped
  categories  ≤ 15 values → one-hot; ≤ 500 values → frequency encoding;
              more → dropped as free text
  ratios      a / b for the most predictive numeric pairs, kept only when the
              ratio predicts the target clearly better than a and b alone.
              Ratios use the values before log/clip transforms, so they are
              inserted ahead of those steps.

Every decision is made on the sample as cleaned so far, and every step is
engine-neutral, so the same pandas / polars / DuckDB / PySpark writers apply.
"""

from __future__ import annotations

import re
from itertools import permutations
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from ..analyzers.target import predictive_power
from ..clean import detectors
from ..codegen.steps import (
    DATE_PARTS, ClipOutliers, DateParts, DropColumns, FrequencyEncode, Log1p, OneHot, Plan, Ratio,
    TextStats,
)

if TYPE_CHECKING:
    from ..report import EDAReport

__all__ = ["plan_features", "time_order_columns"]

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$")
ONEHOT_MAX = 15
FREQ_MAX = 500
RATIO_CANDIDATES = 6        # most predictive numeric columns considered for ratios
RATIO_MAX = 5
RATIO_MIN_GAIN = 0.02       # ratio must beat both inputs by this much predictive power


def plan_features(
    report: "EDAReport",
    base: Plan,
    *,
    target: Optional[str] = None,
    dates: bool = True,
    encode: bool = True,
    ratios: bool = True,
    random_state: int = 42,
) -> Plan:
    """The cleaning plan `base` extended with feature steps."""
    if report.sample is None:
        raise ValueError("This report has no sample attached; re-run flasheda.analyze().")
    target_info = report.target or {}
    target = target or target_info.get("target")
    task = target_info.get("task")

    df = base.apply(report.sample)
    features = [c for c in df.columns if c != target]
    taken = set(df.columns)
    plan = Plan()
    drop: Dict[str, str] = {}

    # Ratios are computed on values before log/clip transforms.
    cut = next((i for i, s in enumerate(base.steps) if isinstance(s, (Log1p, ClipOutliers))),
               len(base.steps))
    before, after = base.steps[:cut], base.steps[cut:]
    ratio_steps: List[Ratio] = []

    def fresh(name: str) -> str:
        while name in taken:
            name += "_"
        taken.add(name)
        return name

    # ── dates ───────────────────────────────────────────────────────────────
    date_cols = []
    if dates:
        for col in features:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                is_text, ts = False, s
            elif _is_iso_text(s):
                is_text, ts = True, pd.to_datetime(s, errors="coerce", format="ISO8601")
            else:
                continue
            parts = [p for p in DATE_PARTS if p != "hour" or (ts.dt.hour.fillna(0) != 0).mean() > 0.01]
            names = tuple(fresh(f"{col}_{p}") for p in parts)
            plan.steps.append(DateParts(col, tuple(parts), names, is_text,
                                        reason="calendar parts are usable by models; raw dates are not"))
            drop[col] = "replaced by date parts"
            date_cols.append(col)

    # ── text ────────────────────────────────────────────────────────────────
    text_info = report.text or {}
    handled = set(date_cols)
    if encode:
        for col in text_info.get("personal_data_columns", []):
            if col in features and col not in handled:
                kinds = ", ".join(text_info["columns"][col]["personal_data"])
                drop[col] = f"personal data ({kinds})"
                handled.add(col)
        for col in text_info.get("free_text_columns", []):
            if col in features and col not in handled and detectors.is_text(df[col]):
                plan.steps.append(TextStats(col, fresh(f"{col}_length"), fresh(f"{col}_words"),
                                            reason="free text: models can use its length and word count"))
                drop[col] = "free text, replaced by length and word count"
                handled.add(col)

    # ── categories ──────────────────────────────────────────────────────────
    if encode:
        for col in features:
            if col in handled or not detectors.is_text(df[col]):
                continue
            counts = df[col].value_counts()
            n_unique = len(counts)
            if n_unique <= 1:
                drop[col] = "a single value"
            elif n_unique <= ONEHOT_MAX:
                values = tuple(str(v) for v in counts.index)
                names = tuple(fresh(_safe(f"{col}_{v}")) for v in values)
                plan.steps.append(OneHot(col, values, names, reason=f"{n_unique} categories"))
                drop[col] = "replaced by one-hot columns"
            elif n_unique <= FREQ_MAX:
                shares = (counts / counts.sum()).round(6)
                mapping = tuple((str(v), float(f)) for v, f in shares.items())
                plan.steps.append(FrequencyEncode(
                    col, mapping, reason=f"{n_unique} categories: too many to one-hot"))
            else:
                drop[col] = f"{n_unique} distinct values in the sample: free text or an identifier"

    # ── ratios (need a target to judge them) ────────────────────────────────
    if ratios and target is not None and task:
        raw = Plan(list(before)).apply(report.sample)
        y = raw[target]
        rng = np.random.default_rng(random_state)
        numeric = [c for c in features if c not in drop and c in raw.columns
                   and pd.api.types.is_numeric_dtype(raw[c]) and not pd.api.types.is_bool_dtype(raw[c])]
        power = {c: predictive_power(raw[c], y, task, rng) or 0.0 for c in numeric}
        top = sorted(numeric, key=lambda c: power[c], reverse=True)[:RATIO_CANDIDATES]
        found = []
        for a, b in permutations(top, 2):
            den = raw[b].astype(float)
            ratio = raw[a].astype(float) / den.where(den != 0)
            gain = (predictive_power(ratio, y, task, rng) or 0.0) - max(power[a], power[b])
            if gain >= RATIO_MIN_GAIN:
                found.append((gain, a, b))
        pairs = set()
        for gain, a, b in sorted(found, reverse=True):
            if len(ratio_steps) == RATIO_MAX or frozenset((a, b)) in pairs:
                continue          # b / a carries the same information as a / b
            pairs.add(frozenset((a, b)))
            ratio_steps.append(Ratio(a, b, fresh(_safe(f"{a}_per_{b}")),
                                     reason=f"predicts the target better than either input (+{gain:.3f} power)"))

    if drop:
        plan.steps.append(DropColumns(tuple(drop), reason="; ".join(f"{c}: {r}" for c, r in drop.items())))
    return Plan(list(before) + ratio_steps + list(after) + plan.steps)


def time_order_columns(plan: Plan, time_columns: List[str]) -> List[str]:
    """Date-part columns that sort rows in time order (for time-based validation)."""
    for step in plan:
        if isinstance(step, DateParts) and step.column in time_columns:
            order = [n for p, n in zip(step.parts, step.names) if p != "dayofweek"]
            return order
    return []


def _is_iso_text(s: pd.Series) -> bool:
    if not detectors.is_text(s):
        return False
    values = s.dropna().astype(str).head(500)
    return bool(values.str.match(ISO_DATE).mean() >= 0.95)


def _safe(name: str) -> str:
    """Column-name friendly: letters, digits and underscores only."""
    return re.sub(r"\W+", "_", str(name)).strip("_") or "col"
