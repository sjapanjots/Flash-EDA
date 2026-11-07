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
  9. distributions   log-transform skewed columns; IQR outliers become a
                     suggestion (clipped only with clip_outliers=True)

The target column is never transformed, only filtered for nulls. Columns in
`protect` are never dropped or changed either (only whole duplicate rows
can still go: pass drop_duplicates=False to keep those too).

Every step gets a risk level (flasheda.codegen.registry) and a confidence:
how sure the sample is that the full data would lead to the same decision.
`approve` decides which steps are applied without asking:

    "confident"  (default) all but high-risk steps with confidence < 0.95
    "all"        every step
    "safe"       only low-risk steps

Skipped steps are kept in plan.skipped and listed in the generated code;
see them all with plan.review().
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import pandas as pd

from ..analyzers import categorical, missing, numeric
from ..clean import detectors
from .registry import CONFIDENT, info, margin_confidence, proportion_confidence
from .steps import (
    ClipOutliers, DropColumns, DropDuplicates, DropNullRows, FillNull, GroupRare, Log1p,
    MissingIndicator, NormalizeText, Plan, ToNumeric,
)

if TYPE_CHECKING:
    from ..report import EDAReport

OUTLIER_CLIP_PCT = 5.0
LEAK_HOLDOUT_SHARE = 0.3       # predictive power is scored on 30% of the sample
APPROVE = ("confident", "all", "safe")


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
    clip_outliers: bool = False,
    approve: str = "confident",
    protect: Sequence[str] = (),
    decimal: Optional[str] = None,
) -> Plan:
    if decimal not in (None, ".", ","):
        raise ValueError("decimal must be None (detect), '.' or ','")
    if approve not in APPROVE:
        raise ValueError(f"approve must be one of {APPROVE}")
    sample = report.sample
    if sample is None:
        raise ValueError("This report has no sample attached; re-run flasheda.analyze().")

    target_info = report.target or {}
    target = target or target_info.get("target")
    if target is not None and target not in sample.columns:
        raise KeyError(f"Target column {target!r} not found.")
    task = target_info.get("task")
    protected = check_protected(protect, sample.columns, target)

    plan = Plan()
    current = lambda: plan.apply(sample)  # noqa: E731 — the sample as cleaned so far
    n = len(sample)

    def add(step) -> None:
        meta = info(step)
        if (approve == "all" or meta.risk == "low"
                or (approve == "confident" and (meta.risk != "high" or step.confidence >= CONFIDENT))):
            plan.steps.append(step)
        elif approve == "confident":
            plan.skipped.append((step, f"{meta.risk} risk ({meta.effect}) and the sample isn't sure: "
                                       f"confidence {step.confidence:.2f} < {CONFIDENT}"))
        else:
            plan.skipped.append((step, f"{meta.risk} risk ({meta.effect}); approve='safe' "
                                       "applies only low-risk steps"))

    # 1 ── drop columns ─────────────────────────────────────────────────────
    drop: Dict[str, str] = {}
    sure: Dict[str, float] = {}          # confidence per dropped column (default 1.0)
    if drop_ids:
        for col in report.categorical.get("likely_id_columns", []):
            drop.setdefault(col, "likely an ID column")
        for col in sample.columns:
            if detectors.is_numeric_id(sample[col], col):
                drop.setdefault(col, "numeric ID column")
    if drop_missing:
        for col in report.missing.get("critical_columns", []):
            if drop.setdefault(col, "more than 50% missing") == "more than 50% missing":
                sure[col] = proportion_confidence(float(sample[col].isna().mean()), 0.5, n)
    for col, s in report.numeric.get("columns", {}).items():
        if s.get("zero_variance"):
            drop.setdefault(col, "constant value")
    if drop_leaks:
        for leak in target_info.get("leakage_columns", []):
            col = leak["column"]
            if col not in drop:
                drop[col] = f"possible target leakage (power {leak['score']})"
                sure[col] = proportion_confidence(leak["score"], target_power_threshold(),
                                                  int(n * LEAK_HOLDOUT_SHARE))
    for col in [target, *protected]:
        drop.pop(col, None)
    confident = {c: r for c, r in drop.items() if sure.get(c, 1.0) >= CONFIDENT}
    if confident:
        add(DropColumns(tuple(confident), reason="; ".join(f"{c}: {r}" for c, r in confident.items())))
    for col in drop:
        if col not in confident:          # judged one by one, so one doubt doesn't block the rest
            add(DropColumns((col,), reason=f"{col}: {drop[col]}", confidence=sure[col]))

    # 2 ── rows without a target ────────────────────────────────────────────
    if target is not None and drop_null_target and sample[target].isna().any():
        pct = round(sample[target].isna().mean() * 100, 2)
        add(DropNullRows(target, reason=f"target is null in {pct}% of sample"))

    features = [c for c in current().columns if c != target and c not in protected]

    # 3 + 4 ── types and text ───────────────────────────────────────────────
    df = current()
    for col in features:
        if fix_numeric_text and detectors.is_numeric_text(df[col]):
            ratio = detectors.numeric_text_ratio(df[col])
            add(ToNumeric(col, reason=f"{ratio:.0%} of values are numbers stored as text",
                          confidence=proportion_confidence(ratio, detectors.NUMERIC_TEXT_MIN_RATIO,
                                                           int(df[col].notna().sum()))))
        elif fix_numeric_text and (fmt := detectors.number_format(df[col], decimal)) is not None:
            if fmt.ambiguous:
                plan.suggestions.append(
                    f"{col}: numbers like {_example(df[col])!r} could use ',' or '.' as the "
                    "decimal mark; pass decimal='.' or decimal=',' to convert them")
            else:
                add(ToNumeric(col, fmt.thousands, fmt.decimal, fmt.strip,
                              reason=f"{fmt.ratio:.0%} of values are numbers written with "
                                     f"separators or symbols",
                              confidence=proportion_confidence(fmt.ratio, detectors.NUMERIC_TEXT_MIN_RATIO,
                                                               int(df[col].notna().sum()))))
        elif normalize_text:
            variants = detectors.text_variants(df[col])
            if variants:
                add(NormalizeText(
                    col, reason=f"{variants[0]} spellings collapse to {variants[1]} after trim/lower-case"))

    # 5 ── duplicates ───────────────────────────────────────────────────────
    df = current()
    dups = int(df.duplicated().sum())
    if drop_duplicates and dups:
        add(DropDuplicates(reason=f"{dups / len(df):.2%} duplicate rows in sample"))

    # 6 ── rare categories ──────────────────────────────────────────────────
    if group_rare:
        df = current()
        for col in features:
            rare = detectors.rare_categories(df[col])
            if rare:
                keep, other = rare
                n_unique = df[col].nunique()
                add(GroupRare(col, keep, other,
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
            add(MissingIndicator(col, name, reason=f"target differs when missing (gap {gap})"))

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
                add(FillNull(col, _number(s["median"]), reason="median of sample"))
            n_col = int(df[col].notna().sum())
            if fix_skew and s.get("skewed") and s["min"] >= 0:
                add(Log1p(col, reason=f"skewness {s['skewness']}",
                          confidence=margin_confidence(abs(s["skewness"]), numeric.SKEW_THRESHOLD,
                                                       math.sqrt(6 / max(n_col, 1)))))
            elif s.get("outlier_pct", 0) > OUTLIER_CLIP_PCT and s["iqr"] > 0:
                lower = round(s["q25"] - 1.5 * s["iqr"], 6)
                upper = round(s["q75"] + 1.5 * s["iqr"], 6)
                if clip_outliers:
                    add(ClipOutliers(col, lower, upper, reason=f"{s['outlier_pct']}% IQR outliers",
                                     confidence=proportion_confidence(
                                         s["outlier_pct"] / 100, OUTLIER_CLIP_PCT / 100, n_col)))
                else:
                    # Outliers are often real (big orders, rare events): flag, don't change.
                    plan.suggestions.append(
                        f"{col}: {s['outlier_pct']}% of values are IQR outliers outside "
                        f"[{lower:g}, {upper:g}]; check them, or pass clip_outliers=True to clip")
        elif detectors.is_text(df[col]) and col in cat_stats and "error" not in cat_stats[col]:
            if fill_nulls and has_nulls:
                add(FillNull(col, cat_stats[col]["mode"], reason="mode of sample"))

    return plan


def _number(x: float):
    return int(x) if float(x).is_integer() else float(x)


def target_power_threshold() -> float:
    from ..analyzers.target import LEAK_THRESHOLD
    return LEAK_THRESHOLD


def check_protected(protect: Sequence[str], columns, target: Optional[str]) -> set:
    if isinstance(protect, str):
        protect = [protect]
    missing = [c for c in protect if c not in columns]
    if missing:
        raise KeyError(f"Protected column(s) not found: {missing}")
    return set(protect)


def _example(s: pd.Series) -> str:
    values = s.dropna().astype(str)
    with_sep = values[values.str.contains(r"[.,]")]
    return str((with_sep if len(with_sep) else values).iloc[0])
