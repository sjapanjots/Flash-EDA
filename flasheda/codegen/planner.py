"""
flasheda.codegen.planner
~~~~~~~~~~~~~~~~~~~~~~~~
Turns EDA findings into a cleaning `Plan`.

The plan is built in phases. After each phase the plan so far is applied
to the sample, so later decisions (fill values, skew, rare categories) are
made on the data as it will look at that point, e.g. a price column
converted from text gets a numeric median.

  1. drop columns    likely IDs, >50% missing, constant, possible leakage
  2. target rows     drop rows where the target is null
  3. fix types       numbers stored as text → numbers
  4. fix text        trim + lower-case columns with spelling variants
  5. duplicates      drop duplicate rows
  6. rare values     group long-tail categories into "other"
  7. missing flags   add <col>_was_missing where missingness predicts the target
  8. fill nulls      numeric median / text mode
  9. distributions   log-transform skewed columns, else clip IQR outliers

The target column is never transformed, only filtered for nulls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

from ..analyzers import categorical, missing, numeric
from ..clean import detectors
from .steps import (
    ClipOutliers, DropColumns, DropDuplicates, DropNullRows, FillNull, GroupRare, Log1p,
    MissingIndicator, NormalizeText, Plan, ToNumeric,
)

if TYPE_CHECKING:
    from ..report import EDAReport

OUTLIER_CLIP_PCT = 5.0


def plan_from_report(
    report: "EDAReport",
    *,
    target: Optional[str] = None,
    drop_ids: bool = True,
    drop_missing: bool = True,
    drop_leaks: bool = True,
    drop_null_target: bool = True,
    fix_numeric_text: bool = True,
    normalize_text: bool = True,
    drop_duplicates: bool = True,
    group_rare: bool = True,
    missing_indicators: bool = True,
    fill_nulls: bool = True,
    fix_skew: bool = True,
    clip_outliers: bool = True,
) -> Plan:
    sample = report.sample
    if sample is None:
        raise ValueError("This report has no sample attached; re-run flasheda.analyze().")

    target_info = report.target or {}
    target = target or target_info.get("target")
    if target is not None and target not in sample.columns:
        raise KeyError(f"Target column {target!r} not found.")
    task = target_info.get("task")

    plan = Plan()
    current = lambda: plan.apply(sample)  # noqa: E731 — the sample as cleaned so far

    # 1 ── drop columns ─────────────────────────────────────────────────────
    drop: Dict[str, str] = {}
    if drop_ids:
        for col in report.categorical.get("likely_id_columns", []):
            drop.setdefault(col, "likely an ID column")
        for col in sample.columns:
            if detectors.is_numeric_id(sample[col], col):
                drop.setdefault(col, "numeric ID column")
    if drop_missing:
        for col in report.missing.get("critical_columns", []):
            drop.setdefault(col, "more than 50% missing")
    for col, s in report.numeric.get("columns", {}).items():
        if s.get("zero_variance"):
            drop.setdefault(col, "constant value")
    if drop_leaks:
        for leak in target_info.get("leakage_columns", []):
            drop.setdefault(leak["column"], f"possible target leakage (power {leak['score']})")
    drop.pop(target, None)
    if drop:
        reasons = "; ".join(f"{c}: {r}" for c, r in drop.items())
        plan.steps.append(DropColumns(tuple(drop), reason=reasons))

    # 2 ── rows without a target ────────────────────────────────────────────
    if target is not None and drop_null_target and sample[target].isna().any():
        pct = round(sample[target].isna().mean() * 100, 2)
        plan.steps.append(DropNullRows(target, reason=f"target is null in {pct}% of sample"))

    features = [c for c in current().columns if c != target]

    # 3 + 4 ── types and text ───────────────────────────────────────────────
    df = current()
    for col in features:
        if fix_numeric_text and detectors.is_numeric_text(df[col]):
            ratio = detectors.numeric_text_ratio(df[col])
            plan.steps.append(ToNumeric(col, reason=f"{ratio:.0%} of values are numbers stored as text"))
        elif normalize_text:
            variants = detectors.text_variants(df[col])
            if variants:
                plan.steps.append(NormalizeText(
                    col, reason=f"{variants[0]} spellings collapse to {variants[1]} after trim/lower-case"))

    # 5 ── duplicates ───────────────────────────────────────────────────────
    df = current()
    dups = int(df.duplicated().sum())
    if drop_duplicates and dups:
        plan.steps.append(DropDuplicates(reason=f"{dups / len(df):.2%} duplicate rows in sample"))

    # 6 ── rare categories ──────────────────────────────────────────────────
    if group_rare:
        df = current()
        for col in features:
            rare = detectors.rare_categories(df[col])
            if rare:
                keep, other = rare
                n_unique = df[col].nunique()
                plan.steps.append(GroupRare(col, keep, other,
                                            reason=f"{n_unique - len(keep)} of {n_unique} values are under 1% each"))

    # 7 ── informative missingness ──────────────────────────────────────────
    df = current()
    if missing_indicators and target is not None and task:
        for col in features:
            gap = detectors.informative_missing(df[col], df[target], task)
            if gap is None:
                continue
            name = f"{col}_was_missing"
            while name in df.columns:
                name += "_"
            plan.steps.append(MissingIndicator(col, name, reason=f"target differs when missing (gap {gap})"))

    # 8 + 9 ── fill, skew, outliers (stats from the sample as cleaned so far) ─
    df = current()
    nulls = missing.analyze(df[features])["per_column"]
    num_stats = numeric.analyze(df[features])["columns"]
    cat_stats = categorical.analyze(df[features])["columns"]
    for col in features:
        has_nulls = nulls[col]["null_count"] > 0
        if col in num_stats and "error" not in num_stats[col]:
            s = num_stats[col]
            if fill_nulls and has_nulls:
                plan.steps.append(FillNull(col, _number(s["median"]), reason="median of sample"))
            if fix_skew and s.get("skewed") and s["min"] >= 0:
                plan.steps.append(Log1p(col, reason=f"skewness {s['skewness']}"))
            elif clip_outliers and s.get("outlier_pct", 0) > OUTLIER_CLIP_PCT and s["iqr"] > 0:
                lower = round(s["q25"] - 1.5 * s["iqr"], 6)
                upper = round(s["q75"] + 1.5 * s["iqr"], 6)
                plan.steps.append(
                    ClipOutliers(col, lower, upper, reason=f"{s['outlier_pct']}% IQR outliers")
                )
        elif detectors.is_text(df[col]) and col in cat_stats and "error" not in cat_stats[col]:
            if fill_nulls and has_nulls:
                plan.steps.append(FillNull(col, cat_stats[col]["mode"], reason="mode of sample"))

    return plan


def _number(x: float):
    return int(x) if float(x).is_integer() else float(x)
