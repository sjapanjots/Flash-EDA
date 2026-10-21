"""
flasheda.compare
~~~~~~~~~~~~~~~~
What changed between two datasets (last month vs this month, train vs
test), measured on fixed-size samples so it takes seconds at any size:

    drift = flasheda.compare("orders_jan.csv", "orders_feb.csv")
    drift.show()

  schema     columns added / removed, type changes
  per column PSI (population stability index) on reference quantile bins or
             categories; KS test for numbers; new categories; null-rate change
  dates, IDs reported as a range shift (they always move), not as drift
  severity   PSI < 0.1 stable · 0.1–0.25 moderate · ≥ 0.25 major (the usual
             credit-risk thresholds); a null rate moving ≥ 10 points is at
             least moderate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..contract.rules import _kind
from ..engine import inspect_source, sample

__all__ = ["DriftReport", "compare", "psi"]

PSI_MODERATE = 0.10
PSI_MAJOR = 0.25
NULL_SHIFT_POINTS = 10.0
BINS = 10
TOP_CATEGORIES = 20
EPS = 1e-4
HIGH_CARDINALITY = 0.5     # unique / rows above this: IDs, free text


@dataclass
class DriftReport:
    rows_reference: int
    rows_current: int
    added_columns: List[str]
    removed_columns: List[str]
    type_changes: List[Dict[str, str]]
    columns: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def drifted(self) -> List[Dict[str, Any]]:
        return [c for c in self.columns if c["severity"] != "stable"]

    @property
    def ok(self) -> bool:
        """No schema changes and no major drift."""
        return (not self.added_columns and not self.removed_columns and not self.type_changes
                and all(c["severity"] != "major" for c in self.columns))

    def __getitem__(self, column: str) -> Dict[str, Any]:
        for c in self.columns:
            if c["column"] == column:
                return c
        raise KeyError(column)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "rows_reference": self.rows_reference,
                "rows_current": self.rows_current, "added_columns": self.added_columns,
                "removed_columns": self.removed_columns, "type_changes": self.type_changes,
                "columns": self.columns}

    def show(self) -> None:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"[bold]Drift[/bold]: {self.rows_reference:,} → {self.rows_current:,} rows  "
                      + ("[green]no major drift[/]" if self.ok else "[red]changes found[/]"))
        if self.added_columns:
            console.print(f"  [yellow]added columns:[/] {', '.join(self.added_columns)}")
        if self.removed_columns:
            console.print(f"  [red]removed columns:[/] {', '.join(self.removed_columns)}")
        for t in self.type_changes:
            console.print(f"  [red]type change:[/] {t['column']}: {t['reference']} → {t['current']}")
        table = Table(box=box.SIMPLE)
        for h in ("column", "PSI", "severity", "null % (ref → cur)", "detail"):
            table.add_column(h)
        colors = {"stable": "green", "moderate": "yellow", "major": "red"}
        for c in self.columns:
            table.add_row(c["column"], f"{c['psi']:.3f}", f"[{colors[c['severity']]}]{c['severity']}[/]",
                          f"{c['null_pct_reference']} → {c['null_pct_current']}", c["detail"])
        console.print(table)

    def __repr__(self) -> str:
        return (f"DriftReport(ok={self.ok}, drifted={len(self.drifted)}/{len(self.columns)}, "
                f"added={len(self.added_columns)}, removed={len(self.removed_columns)})")


def compare(reference: Any, current: Any, *, n: Optional[int] = None,
            random_state: int = 42) -> DriftReport:
    ref_info, cur_info = inspect_source(reference), inspect_source(current)
    ref = sample(reference, n=n, random_state=random_state)
    cur = sample(current, n=n, random_state=random_state)
    ref.columns = [str(c) for c in ref.columns]
    cur.columns = [str(c) for c in cur.columns]

    common = [c for c in ref.columns if c in cur.columns]
    report = DriftReport(
        rows_reference=ref_info.n_rows if ref_info.n_rows is not None else len(ref),
        rows_current=cur_info.n_rows if cur_info.n_rows is not None else len(cur),
        added_columns=[c for c in cur.columns if c not in ref.columns],
        removed_columns=[c for c in ref.columns if c not in cur.columns],
        type_changes=[],
    )
    for col in common:
        k_ref, k_cur = _kind(ref[col]), _kind(cur[col])
        if k_ref != k_cur:
            report.type_changes.append({"column": col, "reference": k_ref, "current": k_cur})
            continue
        report.columns.append(_column_drift(col, ref[col], cur[col], k_ref))
    report.columns.sort(key=lambda c: c["psi"], reverse=True)
    return report


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population stability index between two proportion vectors."""
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _column_drift(col: str, ref: pd.Series, cur: pd.Series, kind: str) -> Dict[str, Any]:
    null_ref = round(float(ref.isna().mean() * 100), 2)
    null_cur = round(float(cur.isna().mean() * 100), 2)
    out: Dict[str, Any] = {"column": col, "kind": kind,
                           "null_pct_reference": null_ref, "null_pct_current": null_cur}

    if kind == "datetime":
        # Dates move forward by nature, so they're reported as a range shift,
        # not as drift; only their null rate can raise the severity.
        r = pd.to_datetime(ref, errors="coerce", format="mixed").dropna()
        c = pd.to_datetime(cur, errors="coerce", format="mixed").dropna()
        value = 0.0
        out["detail"] = (f"{r.min():%Y-%m-%d}…{r.max():%Y-%m-%d} → {c.min():%Y-%m-%d}…{c.max():%Y-%m-%d}"
                         if len(r) and len(c) else "")
    elif kind == "integer" and _is_id_like(ref):
        # New IDs always differ from old ones: report the range, not drift.
        r, c = ref.dropna(), cur.dropna()
        value = 0.0
        out["detail"] = (f"ID-like: {r.min():,}…{r.max():,} → {c.min():,}…{c.max():,}"
                         if len(r) and len(c) else "ID-like")
    elif kind in ("number", "integer"):
        r = pd.to_numeric(ref, errors="coerce")
        c = pd.to_numeric(cur, errors="coerce")
        value = _numeric_psi(r, c)
        out.update(_numeric_details(r.dropna(), c.dropna()))
    else:
        value, new_share, new_values = _categorical_psi(ref.astype("string"), cur.astype("string"))
        values = ref.dropna()
        if len(values) and values.nunique() / len(values) > HIGH_CARDINALITY:
            out["detail"] = "mostly unique values: compared by most common values only"
        else:
            out["new_values_pct"] = round(new_share * 100, 2)
            out["new_values"] = new_values
            out["detail"] = (f"{out['new_values_pct']}% new values, e.g. {', '.join(new_values[:3])}"
                             if new_values else "")
    out["psi"] = round(value, 4)

    severity = "major" if value >= PSI_MAJOR else "moderate" if value >= PSI_MODERATE else "stable"
    if severity == "stable" and abs(null_cur - null_ref) >= NULL_SHIFT_POINTS:
        severity = "moderate"
    out["severity"] = severity
    return out


def _is_id_like(s: pd.Series) -> bool:
    values = s.dropna()
    return len(values) > 0 and values.nunique() / len(values) > 0.95


def _numeric_psi(ref: pd.Series, cur: pd.Series) -> float:
    r, c = ref.dropna().to_numpy(), cur.dropna().to_numpy()
    if not len(r) or not len(c):
        return 0.0
    edges = np.unique(np.quantile(r, np.linspace(0, 1, BINS + 1)))
    if len(edges) < 2:                                   # constant reference
        return psi([1.0, 0.0], [np.mean(c == edges[0]), np.mean(c != edges[0])])
    edges[0], edges[-1] = -np.inf, np.inf
    # missing values get their own bin, so a jump in nulls counts as drift
    e = np.append(np.histogram(r, edges)[0] / len(ref), ref.isna().mean())
    a = np.append(np.histogram(c, edges)[0] / len(cur), cur.isna().mean())
    return psi(e, a)


def _numeric_details(r: pd.Series, c: pd.Series) -> Dict[str, Any]:
    from scipy.stats import ks_2samp

    if not len(r) or not len(c):
        return {"detail": "no values to compare"}
    ks = ks_2samp(r, c)
    std = float(r.std()) or 1.0
    shift = (float(c.median()) - float(r.median())) / std
    return {
        "median_reference": float(r.median()),
        "median_current": float(c.median()),
        "median_shift_std": round(shift, 3),
        "ks_statistic": round(float(ks.statistic), 4),
        "ks_pvalue": float(ks.pvalue),
        "detail": f"median {float(r.median()):.4g} → {float(c.median()):.4g} ({shift:+.2f} std)",
    }


def _categorical_psi(ref: pd.Series, cur: pd.Series):
    ref_counts = ref.fillna("<null>").value_counts(normalize=True)
    cur_counts = cur.fillna("<null>").value_counts(normalize=True)
    top = list(ref_counts.index[:TOP_CATEGORIES])
    if "<null>" not in top:
        top.append("<null>")
    e = [ref_counts.get(v, 0.0) for v in top] + [1 - ref_counts.reindex(top).fillna(0).sum()]
    a = [cur_counts.get(v, 0.0) for v in top] + [1 - cur_counts.reindex(top).fillna(0).sum()]
    seen = set(ref_counts.index) | {"<null>"}      # nulls are reported as null %, not new values
    new = cur_counts[[v not in seen for v in cur_counts.index]]
    return psi(e, a), float(new.sum()), [str(v) for v in new.index[:10]]
