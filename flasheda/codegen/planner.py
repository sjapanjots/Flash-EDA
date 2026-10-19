"""
flasheda.codegen.planner
~~~~~~~~~~~~~~~~~~~~~~~~
Turns EDA findings into a starter cleaning `Plan`.

Rules (each step records why it was added):
  - drop likely-ID, mostly-missing and constant columns
  - drop duplicate rows if any were seen in the sample
  - fill numeric nulls with the median, text nulls with the mode
  - log1p heavily skewed non-negative columns
  - otherwise clip columns with many IQR outliers to the IQR fences
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .steps import ClipOutliers, DropColumns, DropDuplicates, FillNull, Log1p, Plan

if TYPE_CHECKING:
    from ..report import EDAReport

OUTLIER_CLIP_PCT = 5.0


def plan_from_report(
    report: "EDAReport",
    *,
    drop_ids: bool = True,
    drop_missing: bool = True,
    drop_duplicates: bool = True,
    fill_nulls: bool = True,
    fix_skew: bool = True,
    clip_outliers: bool = True,
) -> Plan:
    plan = Plan()
    col_types = report.overview.get("col_types", {})
    num_stats = report.numeric.get("columns", {})
    cat_stats = report.categorical.get("columns", {})
    per_col_missing = report.missing.get("per_column", {})

    drop: dict = {}
    if drop_ids:
        for col in report.categorical.get("likely_id_columns", []):
            drop.setdefault(col, "likely an ID column")
    if drop_missing:
        for col in report.missing.get("critical_columns", []):
            drop.setdefault(col, "more than 50% missing")
    for col, s in num_stats.items():
        if s.get("zero_variance"):
            drop.setdefault(col, "constant value")
    if drop:
        reasons = "; ".join(f"{c}: {r}" for c, r in drop.items())
        plan.steps.append(DropColumns(tuple(drop), reason=reasons))

    if drop_duplicates and report.overview.get("duplicate_rows_in_sample", 0) > 0:
        pct = report.overview.get("duplicate_pct_in_sample", 0)
        plan.steps.append(DropDuplicates(reason=f"{pct}% duplicate rows in sample"))

    for col in report.overview.get("column_names", []):
        if col in drop:
            continue
        has_nulls = per_col_missing.get(col, {}).get("null_count", 0) > 0
        ctype = col_types.get(col)

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

        elif ctype == "string/object" and col in cat_stats and "error" not in cat_stats[col]:
            if fill_nulls and has_nulls:
                plan.steps.append(FillNull(col, cat_stats[col]["mode"], reason="mode of sample"))

    return plan


def _number(x: float):
    return int(x) if float(x).is_integer() else float(x)
