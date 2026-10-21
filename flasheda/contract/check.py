"""
flasheda.contract.check
~~~~~~~~~~~~~~~~~~~~~~~
Checks the *full* data against a contract.

Files are checked with one DuckDB query (a single streaming pass, works on
files larger than memory); DataFrames, or files when DuckDB isn't
installed, are checked with pandas. Both compute the same statistics, and
`evaluate()` turns them into pass/fail outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..engine.backends import is_available
from ..engine.source import CSV_EXTS, DUCKDB_CSV_TYPES, PARQUET_EXTS, _is_polars_frame, csv_sep
from .rules import ColumnRule, Contract, invalid_share


@dataclass
class Outcome:
    column: Optional[str]
    rule: str
    passed: bool
    observed: Any
    expected: Any

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        where = f"{self.column}: " if self.column else ""
        return f"[{mark}] {where}{self.rule}: observed {self.observed}, expected {self.expected}"


@dataclass
class CheckResult:
    rows: int
    outcomes: List[Outcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(o.passed for o in self.outcomes)

    @property
    def failures(self) -> List[Outcome]:
        return [o for o in self.outcomes if not o.passed]

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "rows": self.rows,
                "outcomes": [o.__dict__ for o in self.outcomes]}

    def show(self, all_checks: bool = False) -> None:
        from rich.console import Console
        console = Console()
        status = "[bold green]PASSED[/]" if self.ok else f"[bold red]FAILED[/] ({len(self.failures)} checks)"
        console.print(f"Contract check on {self.rows:,} rows: {status}")
        for o in (self.outcomes if all_checks else self.failures):
            console.print(f"  {'[green]✓[/]' if o.passed else '[red]✗[/]'} {o}")

    def __repr__(self) -> str:
        return f"CheckResult(ok={self.ok}, rows={self.rows:,}, failures={len(self.failures)})"


def check(source: Any, contract: Contract) -> CheckResult:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if is_available("duckdb"):
            stats, columns, rows = stats_duckdb(path, contract)
        else:
            stats, columns, rows = stats_pandas(_read(path), contract)
    elif isinstance(source, pd.DataFrame):
        stats, columns, rows = stats_pandas(source, contract)
    elif _is_polars_frame(source):
        stats, columns, rows = stats_pandas(source.to_pandas(), contract)
    else:
        raise TypeError(f"source must be a DataFrame or file path, got {type(source)}")
    return evaluate(contract, stats, columns, rows)


# ── statistics ───────────────────────────────────────────────────────────────

def stats_pandas(df: pd.DataFrame, contract: Contract):
    rows = len(df)
    columns = [str(c) for c in df.columns]
    df = df.set_axis(columns, axis=1)
    stats: Dict[str, Dict[str, Any]] = {}
    for rule in contract.columns:
        if rule.name not in df.columns:
            continue
        s = df[rule.name]
        values = s.dropna()
        st: Dict[str, Any] = {"nulls": int(s.isna().sum()), "non_null": int(len(values))}
        st["invalid"] = int(round(invalid_share(values, rule.kind) * len(values)))
        if rule.min is not None or rule.max is not None:
            x = pd.to_numeric(values, errors="coerce").dropna() if len(values) else values
            st["min"] = float(x.min()) if len(x) else None
            st["max"] = float(x.max()) if len(x) else None
        if rule.allowed is not None:
            st["unexpected"] = int((~values.astype(str).isin(rule.allowed)).sum())
        if rule.unique:
            st["distinct"] = int(values.nunique())
        stats[rule.name] = st
    return stats, columns, rows


def stats_duckdb(path: Path, contract: Contract):
    import duckdb

    con = duckdb.connect()
    scan, params = _scan(path)
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {scan}", params).fetchall()]
    present = [r for r in contract.columns if r.name in columns]

    exprs = ["count(*)"]
    keys: List[tuple] = []
    for rule in present:
        c = _ident(rule.name)
        txt = f"CAST({c} AS VARCHAR)"
        parts = {
            "nulls": f"count(*) - count({c})",
            "non_null": f"count({c})",
            "invalid": _invalid_sql(c, txt, rule.kind),
        }
        if rule.min is not None or rule.max is not None:
            parts["min"] = f"min(TRY_CAST({c} AS DOUBLE))"
            parts["max"] = f"max(TRY_CAST({c} AS DOUBLE))"
        if rule.allowed is not None:
            values = ", ".join(_lit(v) for v in rule.allowed)
            parts["unexpected"] = f"count({c}) FILTER (WHERE {txt} NOT IN ({values}))"
        if rule.unique:
            parts["distinct"] = f"count(DISTINCT {c})"
        for key, expr in parts.items():
            keys.append((rule.name, key))
            exprs.append(expr)

    row = con.execute(f"SELECT {', '.join(exprs)} FROM {scan}", params).fetchone()
    stats: Dict[str, Dict[str, Any]] = {r.name: {} for r in present}
    for (name, key), value in zip(keys, row[1:]):
        stats[name][key] = float(value) if key in ("min", "max") and value is not None else value
    return stats, columns, int(row[0])


def _scan(path: Path):
    ext = path.suffix.lower()
    if ext in PARQUET_EXTS:
        return "read_parquet(?)", [str(path)]
    if ext in CSV_EXTS:
        return f"read_csv(?, delim = ?, header = true, {DUCKDB_CSV_TYPES})", [str(path), csv_sep(path)]
    raise ValueError(f"Unsupported file type: {ext}. Use .csv, .tsv or .parquet")


def _invalid_sql(c: str, txt: str, kind: str) -> str:
    if kind in ("number", "integer"):
        return f"count({c}) - count(TRY_CAST(trim({txt}) AS DOUBLE))"
    if kind == "datetime":
        return f"count({c}) - count(TRY_CAST({txt} AS TIMESTAMP))"
    if kind == "bool":
        return f"count({c}) FILTER (WHERE lower(trim({txt})) NOT IN ('true', 'false', '1', '0'))"
    return "0"


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in PARQUET_EXTS:
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=csv_sep(path), low_memory=False)


def _ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


# ── rules → outcomes ─────────────────────────────────────────────────────────

def evaluate(contract: Contract, stats, columns: List[str], rows: int) -> CheckResult:
    result = CheckResult(rows)
    add = result.outcomes.append

    expected = [r.name for r in contract.columns]
    missing = [c for c in expected if c not in columns]
    add(Outcome(None, "required columns", not missing,
                f"missing {missing}" if missing else "all present", f"{len(expected)} columns"))
    extra = [c for c in columns if c not in expected]
    if extra and not contract.allow_extra_columns:
        add(Outcome(None, "no extra columns", False, extra, "none"))

    for rule in contract.columns:
        st = stats.get(rule.name)
        if st is None:
            continue
        _evaluate_column(rule, st, rows, add)
    return result


def _evaluate_column(rule: ColumnRule, st: Dict[str, Any], rows: int, add) -> None:
    name = rule.name
    non_null = st["non_null"]
    if rule.kind == "text":
        pct = round(st["nulls"] / rows * 100, 2) if rows else 0.0
        add(Outcome(name, "null %", pct <= rule.max_null_pct, pct, f"≤ {rule.max_null_pct}"))
    else:
        # Readers disagree on tokens like "N/A" (null in pandas, text in DuckDB),
        # so nulls and unparseable values are judged together.
        pct = round((st["nulls"] + st["invalid"]) / rows * 100, 2) if rows else 0.0
        limit = round(rule.max_null_pct + rule.max_invalid_pct, 2)
        add(Outcome(name, "null or unparseable %", pct <= limit, pct, f"≤ {limit}"))

    if rule.kind != "text":
        bad = round(st["invalid"] / non_null * 100, 2) if non_null else 0.0
        add(Outcome(name, f"values parse as {rule.kind} (% invalid)",
                    bad <= rule.max_invalid_pct, bad, f"≤ {rule.max_invalid_pct}"))
    if rule.min is not None and st.get("min") is not None:
        add(Outcome(name, "minimum", st["min"] >= rule.min, st["min"], f"≥ {rule.min}"))
    if rule.max is not None and st.get("max") is not None:
        add(Outcome(name, "maximum", st["max"] <= rule.max, st["max"], f"≤ {rule.max}"))
    if rule.allowed is not None:
        pct = round(st["unexpected"] / non_null * 100, 2) if non_null else 0.0
        add(Outcome(name, "unexpected values %", pct <= rule.max_unexpected_pct, pct,
                    f"≤ {rule.max_unexpected_pct}"))
    if rule.unique:
        dupes = non_null - st["distinct"]
        add(Outcome(name, "duplicate values", dupes == 0, dupes, 0))
