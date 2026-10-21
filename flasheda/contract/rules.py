"""
flasheda.contract.rules
~~~~~~~~~~~~~~~~~~~~~~~
A data contract: what each column should look like, learned from the
sample with tolerances, so new data that looks like the old data passes
and real breakage (missing columns, new junk values, a column suddenly
half empty, impossible values) fails.

  kind        number / integer / bool / datetime / text, with a small
              allowance for values that don't parse ("N/A" in a number column)
  nulls       max null % = observed + max(1 point, 50% of observed)
  range       plausible bounds: 1st–99th percentile widened by 10× their
              spread, which tolerates long tails but catches unit changes
              (×100, ×1000); non-negative columns stay non-negative; IDs
              have no range
  missing     for typed columns, nulls and unparseable values are counted
              together, because readers disagree ("N/A" is null in pandas,
              text in DuckDB)
  values      allowed set for low-cardinality text, up to 1% unexpected
  unique      for ID-like columns that are unique in the sample
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..clean import detectors

if TYPE_CHECKING:
    from ..report import EDAReport

KINDS = ("number", "integer", "bool", "datetime", "text")
ALLOWED_MAX_VALUES = 20
UNEXPECTED_MAX_PCT = 1.0
RANGE_WIDTH = 10                 # bounds sit this many 1%–99% spreads outside them
INVALID_MARGIN_PCT = 0.5


@dataclass(frozen=True)
class ColumnRule:
    name: str
    kind: str
    max_null_pct: float
    max_invalid_pct: float = INVALID_MARGIN_PCT
    min: Optional[float] = None
    max: Optional[float] = None
    allowed: Optional[tuple] = None
    max_unexpected_pct: float = UNEXPECTED_MAX_PCT
    unique: bool = False

    def describe(self) -> str:
        parts = [self.kind, f"≤ {self.max_null_pct}% null"]
        if self.min is not None or self.max is not None:
            parts.append(f"in [{self.min}, {self.max}]")
        if self.allowed is not None:
            parts.append(f"one of {len(self.allowed)} values (≤ {self.max_unexpected_pct}% others)")
        if self.unique:
            parts.append("unique")
        return f"{self.name}: " + ", ".join(parts)


@dataclass
class Contract:
    columns: List[ColumnRule]
    allow_extra_columns: bool = True
    source: Optional[str] = None
    version: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> ColumnRule:
        for rule in self.columns:
            if rule.name == name:
                return rule
        raise KeyError(name)

    def describe(self) -> List[str]:
        return [r.describe() for r in self.columns]

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for col in d["columns"]:
            if col["allowed"] is not None:
                col["allowed"] = list(col["allowed"])
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Contract":
        cols = [ColumnRule(**{**c, "allowed": tuple(c["allowed"]) if c.get("allowed") is not None else None})
                for c in d["columns"]]
        return cls(cols, d.get("allow_extra_columns", True), d.get("source"),
                   d.get("version", 1), d.get("meta", {}))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Contract":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ── code export ─────────────────────────────────────────────────────────
    def script(self, source: Optional[str] = None) -> str:
        """Standalone DuckDB check script (exits 1 on failure); see contract.script."""
        from .script import render_check_script
        return render_check_script(self, source)

    def to_pandera(self) -> str:
        """pandera schema code with the same rules."""
        from .script import render_pandera
        return render_pandera(self)

    # ── checking ────────────────────────────────────────────────────────────
    def check(self, source: Any):
        """Check the *full* `source` (DataFrame or CSV/Parquet path) against this contract."""
        from .check import check
        return check(source, self)


def infer_contract(report: "EDAReport", allow_extra_columns: bool = True) -> Contract:
    """Learn a contract from the report's sample."""
    sample = report.sample
    if sample is None:
        raise ValueError("This report has no sample attached; re-run flasheda.analyze().")
    likely_ids = set(report.categorical.get("likely_id_columns", []))
    rules = [_infer_rule(sample[c], str(c), likely_ids) for c in sample.columns]
    return Contract(rules, allow_extra_columns, report.source,
                    meta={"learned_from_rows": int(len(sample)),
                          "source_rows": int(report.original_shape[0])})


def _infer_rule(s: pd.Series, name: str, likely_ids: set) -> ColumnRule:
    n = len(s)
    null_pct = float(s.isna().mean() * 100) if n else 0.0
    max_null = round(min(100.0, null_pct + max(1.0, null_pct * 0.5)), 2)
    values = s.dropna()
    kind = _kind(s)
    invalid = invalid_share(values, kind) * 100
    rule = dict(name=name, kind=kind, max_null_pct=max_null,
                max_invalid_pct=round(invalid + INVALID_MARGIN_PCT, 2))

    is_id = name in likely_ids or detectors.is_numeric_id(s, name)
    if kind in ("number", "integer") and len(values) and not is_id:   # IDs grow; no range
        x = pd.to_numeric(values.astype(str).str.strip(), errors="coerce").dropna() \
            if not pd.api.types.is_numeric_dtype(values) else values.astype(float)
        lo, hi = np.percentile(x, [1, 99])
        spread = max(hi - lo, abs(hi) * 0.1, 1e-9)
        low, high = lo - RANGE_WIDTH * spread, hi + RANGE_WIDTH * spread
        if x.min() >= 0:
            low = max(low, 0.0)
        rule.update(min=_round(low), max=_round(high))

    if is_id and len(values) and values.nunique() == len(values):
        rule["unique"] = True
    elif kind == "text" and 0 < values.nunique() <= ALLOWED_MAX_VALUES:
        rule["allowed"] = tuple(sorted(str(v) for v in values.unique()))
    return ColumnRule(**rule)


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s) or detectors.looks_like_datetime(s):
        return "datetime"
    values = s.dropna()
    if len(values) and pd.api.types.infer_dtype(values, skipna=True) == "boolean":
        return "bool"
    if len(values) and values.astype(str).str.strip().str.lower().isin(["true", "false"]).all():
        return "bool"           # "True"/"False" text, e.g. booleans read from CSV
    if detectors.is_numeric_text(s):
        return "number"         # numbers stored as text, with a little junk ("N/A")
    return "text"


def invalid_share(values: pd.Series, kind: str) -> float:
    """Share of non-null values that don't parse as `kind`."""
    if not len(values) or kind == "text":
        return 0.0
    if kind in ("number", "integer"):
        if pd.api.types.is_numeric_dtype(values):
            return 0.0
        return float(pd.to_numeric(values.astype(str).str.strip(), errors="coerce").isna().mean())
    if kind == "datetime":
        if pd.api.types.is_datetime64_any_dtype(values):
            return 0.0
        return float(pd.to_datetime(values.astype(str), errors="coerce", format="mixed").isna().mean())
    if kind == "bool":
        if pd.api.types.is_bool_dtype(values):
            return 0.0
        ok = values.astype(str).str.strip().str.lower().isin(["true", "false", "1", "0"])
        return float((~ok).mean())
    return 0.0


def _round(x: float) -> float:
    return float(f"{x:.6g}")
