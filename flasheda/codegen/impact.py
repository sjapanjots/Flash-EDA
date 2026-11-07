"""
flasheda.codegen.impact
~~~~~~~~~~~~~~~~~~~~~~~
A dry run of a cleaning plan on the *full* data, before you run it for real:

    impact = report.plan().impact("orders.parquet")
    impact.show()

Reports rows removed, columns dropped and added, and per changed column the
null and distinct counts before and after. Files are scanned with DuckDB
(the plan's own SQL, out of core; distinct counts are approximate);
DataFrames use the pandas reference implementation (exact). Nothing is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from .steps import COLUMN_STEPS, Plan, inputs, outputs


@dataclass
class Impact:
    rows_before: int
    rows_after: int
    dropped_columns: List[str]
    added_columns: List[str]
    columns: pd.DataFrame            # column, nulls/distinct before and after
    engine: str                      # "duckdb" (files) or "pandas" (DataFrames)
    exact: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    def to_dict(self) -> Dict[str, Any]:
        return {"rows_before": self.rows_before, "rows_after": self.rows_after,
                "dropped_columns": self.dropped_columns, "added_columns": self.added_columns,
                "columns": self.columns.to_dict("records"), "engine": self.engine,
                "exact": self.exact}

    def show(self) -> None:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        pct = self.rows_removed / self.rows_before * 100 if self.rows_before else 0.0
        console.print(f"[bold]Full-data dry run ({self.engine})[/bold]: "
                      f"{self.rows_before:,} → {self.rows_after:,} rows "
                      f"({self.rows_removed:,} removed, {pct:.2f}%)")
        if self.dropped_columns:
            console.print(f"  dropped: {', '.join(self.dropped_columns)}")
        if self.added_columns:
            console.print(f"  added:   {', '.join(self.added_columns)}")
        distinct = "distinct" + ("" if self.exact else " (approx.)")
        table = Table(box=box.SIMPLE)
        for col in ("column", "nulls before", "nulls after", f"{distinct} before", f"{distinct} after"):
            table.add_column(col)
        fmt = lambda v: "—" if pd.isna(v) else f"{int(v):,}"    # noqa: E731
        for r in self.columns.itertuples(index=False):
            table.add_row(str(r.column), fmt(r.nulls_before), fmt(r.nulls_after),
                          fmt(r.distinct_before), fmt(r.distinct_after))
        if len(self.columns):
            console.print(table)

    def __repr__(self) -> str:
        return (f"Impact(rows {self.rows_before:,} → {self.rows_after:,}, "
                f"dropped={self.dropped_columns}, added={self.added_columns})")


def impact(plan: Plan, source: Union[str, Path, pd.DataFrame]) -> Impact:
    if isinstance(source, pd.DataFrame):
        before, after = source, plan.apply(source)
        b, a = _pandas_stats(before), _pandas_stats(after)
        return _build(plan, list(before.columns), list(after.columns), b, a, "pandas", True)
    if isinstance(source, (str, Path)):
        return _duckdb_impact(plan, Path(source))
    raise TypeError("impact() takes a file path (CSV/TSV/Parquet) or a pandas DataFrame.")


def _touched(plan: Plan) -> List[str]:
    cols: List[str] = []
    for step in plan.steps:
        if isinstance(step, COLUMN_STEPS):
            for c in list(inputs(step)) + list(outputs(step)):
                if c not in cols:
                    cols.append(c)
    return cols


def _build(plan, cols_before, cols_after, b, a, engine, exact) -> Impact:
    dropped = [c for c in cols_before if c not in cols_after]
    added = [c for c in cols_after if c not in cols_before]
    rows = []
    for col in _touched(plan):
        if col in dropped:
            continue
        rows.append({
            "column": col,
            "nulls_before": b["nulls"].get(col), "nulls_after": a["nulls"].get(col),
            "distinct_before": b["distinct"].get(col), "distinct_after": a["distinct"].get(col),
        })
    table = pd.DataFrame(rows, columns=["column", "nulls_before", "nulls_after",
                                        "distinct_before", "distinct_after"])
    return Impact(b["rows"], a["rows"], dropped, added, table, engine, exact)


def _pandas_stats(df: pd.DataFrame) -> Dict[str, Any]:
    return {"rows": len(df),
            "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
            "distinct": {c: int(df[c].nunique()) for c in df.columns}}


def _duckdb_impact(plan: Plan, path: Path) -> Impact:
    import duckdb

    from ..engine.source import DUCKDB_CSV_TYPES, PARQUET_EXTS, csv_sep
    from .writers.duckdb_writer import DuckDBWriter, ident

    if not path.exists():
        raise FileNotFoundError(path)
    con = duckdb.connect()
    con.execute("CREATE TEMP VIEW _fe_before AS SELECT * FROM " + (
        f"read_parquet('{_q(path)}')" if path.suffix.lower() in PARQUET_EXTS else
        f"read_csv('{_q(path)}', delim = '{_q(csv_sep(path))}', header = true, {DUCKDB_CSV_TYPES})"))
    con.execute("CREATE TEMP VIEW _fe_after AS " +
                DuckDBWriter().build_sql(plan).replace("{table}", "_fe_before"))

    def stats(view: str) -> Dict[str, Any]:
        cols = [r[0] for r in con.execute(f"DESCRIBE {view}").fetchall()]
        exprs = ["count(*)"]
        for c in cols:
            exprs += [f"count(*) - count({ident(c)})", f"approx_count_distinct({ident(c)})"]
        row = con.execute(f"SELECT {', '.join(exprs)} FROM {view}").fetchone()
        return {"cols": cols, "rows": int(row[0]),
                "nulls": {c: int(row[1 + 2 * i]) for i, c in enumerate(cols)},
                "distinct": {c: int(row[2 + 2 * i]) for i, c in enumerate(cols)}}

    b, a = stats("_fe_before"), stats("_fe_after")
    return _build(plan, b["cols"], a["cols"], b, a, "duckdb", False)


def _q(value) -> str:
    return str(value).replace("'", "''")
