"""
flasheda.engine.source
~~~~~~~~~~~~~~~~~~~~~~
Cheap metadata about a data source (row count, columns, format)
without loading it into memory.

  Parquet → row count read from file metadata (no data scan)
  CSV/TSV → native DuckDB count when available, otherwise a chunked count
  DataFrame (pandas or polars) → shape directly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from .backends import has_module, is_available

CSV_EXTS = (".csv", ".tsv", ".txt")

# DuckDB's CSV sniffer would turn "yes"/"no" into booleans and strings into
# dates, which pandas and polars don't. Limiting the candidates keeps every
# engine reading the same values.
DUCKDB_CSV_TYPES = "auto_type_candidates = ['BIGINT', 'DOUBLE', 'VARCHAR']"
PARQUET_EXTS = (".parquet", ".pq")


@dataclass
class SourceInfo:
    kind: str                       # "pandas" | "polars" | "csv" | "parquet"
    n_rows: Optional[int]
    n_cols: int
    columns: List[str] = field(default_factory=list)
    path: Optional[Path] = None
    sep: str = ","

    @property
    def is_file(self) -> bool:
        return self.path is not None

    @property
    def shape(self) -> tuple:
        return (self.n_rows if self.n_rows is not None else -1, self.n_cols)

    def describe(self) -> str:
        return str(self.path) if self.path else f"<in-memory {self.kind} DataFrame>"


def inspect_source(source: Any) -> SourceInfo:
    if isinstance(source, pd.DataFrame):
        return SourceInfo("pandas", len(source), source.shape[1], [str(c) for c in source.columns])

    if _is_polars_frame(source):
        return SourceInfo("polars", source.height, source.width, list(source.columns))

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        ext = path.suffix.lower()
        if ext in PARQUET_EXTS:
            return _inspect_parquet(path)
        if ext in CSV_EXTS:
            return _inspect_csv(path)
        raise ValueError(f"Unsupported file type: {ext}. Use .csv, .tsv or .parquet")

    raise TypeError(f"source must be a DataFrame or file path, got {type(source)}")


def csv_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in (".tsv", ".txt") else ","


def _inspect_parquet(path: Path) -> SourceInfo:
    if has_module("pyarrow"):
        import pyarrow.parquet as pq
        meta = pq.ParquetFile(path).metadata
        cols = [meta.schema.column(i).name for i in range(meta.num_columns)]
        return SourceInfo("parquet", meta.num_rows, len(cols), cols, path)
    if is_available("duckdb"):
        import duckdb
        con = duckdb.connect()
        rel = con.read_parquet(str(path))
        n = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
        return SourceInfo("parquet", n, len(rel.columns), list(rel.columns), path)
    df = pd.read_parquet(path)
    return SourceInfo("parquet", len(df), df.shape[1], [str(c) for c in df.columns], path)


def _inspect_csv(path: Path) -> SourceInfo:
    sep = csv_sep(path)
    header = pd.read_csv(path, sep=sep, nrows=0)
    cols = [str(c) for c in header.columns]

    if is_available("duckdb"):
        import duckdb
        n = duckdb.connect().execute(
            "SELECT count(*) FROM read_csv(?, delim=?, header=true)", [str(path), sep]
        ).fetchone()[0]
    else:
        n = sum(len(chunk) for chunk in pd.read_csv(path, sep=sep, usecols=[0], chunksize=500_000))

    return SourceInfo("csv", int(n), len(cols), cols, path, sep)


def _is_polars_frame(obj: Any) -> bool:
    return type(obj).__module__.startswith("polars") and hasattr(obj, "to_pandas")
