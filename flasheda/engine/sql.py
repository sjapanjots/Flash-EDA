"""
flasheda.engine.sql
~~~~~~~~~~~~~~~~~~~
Database tables and queries as sources, through SQLAlchemy:

    src = flasheda.sql("postgresql://user:pass@host/db", table="orders")
    flasheda.analyze(src, target="churn")

Row counts and samples are computed inside the database, so only the
sample (5,000 rows by default) is transferred:

  postgresql   TABLESAMPLE BERNOULLI(p) REPEATABLE(seed), then LIMIT n
  duckdb       USING SAMPLE reservoir(n ROWS) REPEATABLE(seed)
  others       ORDER BY random() LIMIT n   (full scan in the database)

Passwords never appear in reports or generated code: `display` masks
them, and generated scripts read the URL from an environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd

URL_ENV_VAR = "DATABASE_URL"


@dataclass(frozen=True)
class SqlSource:
    url: str
    table: Optional[str] = None
    query: Optional[str] = None

    def __post_init__(self):
        if (self.table is None) == (self.query is None):
            raise ValueError("Pass exactly one of table= or query=.")

    # ── description ─────────────────────────────────────────────────────────
    @property
    def dialect(self) -> str:
        return self._url().get_backend_name()

    @property
    def display(self) -> str:
        """URL with the password masked, plus the table or a short query."""
        what = f"table {self.table}" if self.table else f"query {self.query[:60]!r}"
        return f"{self._url().render_as_string(hide_password=True)} ({what})"

    def from_sql(self) -> str:
        """The FROM clause: a quoted table name or the query as a subquery."""
        if self.table:
            # the dialect class quotes names without loading the database driver
            preparer = self._url().get_dialect()().identifier_preparer
            return ".".join(preparer.quote(part) for part in self.table.split("."))
        return f"({self.query.rstrip().rstrip(';')}) AS flasheda_source"

    # ── data access ─────────────────────────────────────────────────────────
    def count(self) -> int:
        return int(self._scalar(f"SELECT count(*) FROM {self.from_sql()}"))

    def columns(self) -> List[str]:
        return list(self.read(f"SELECT * FROM {self.from_sql()} LIMIT 0").columns)

    def read(self, sql: str) -> pd.DataFrame:
        from sqlalchemy import text
        with self._engine().connect() as con:
            return pd.read_sql(text(sql), con)

    def sample(self, n: int, random_state: int = 42, total: Optional[int] = None) -> pd.DataFrame:
        src = self.from_sql()
        dialect = self.dialect
        if dialect == "duckdb":
            sql = f"SELECT * FROM {src} USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE ({int(random_state)})"
        elif dialect == "postgresql" and self.table:
            total = self.count() if total is None else total
            pct = min(100.0, 100.0 * 1.5 * n / max(total, 1))
            sql = (f"SELECT * FROM {src} TABLESAMPLE BERNOULLI ({pct:.6f}) "
                   f"REPEATABLE ({int(random_state)}) LIMIT {int(n)}")
        else:
            rand = "rand()" if dialect in ("mysql", "mariadb") else "random()"
            sql = f"SELECT * FROM {src} ORDER BY {rand} LIMIT {int(n)}"
        return self.read(sql).reset_index(drop=True)

    def to_parquet(self, path: Union[str, Path], chunksize: int = 100_000) -> Path:
        """Stream the whole table/query to a Parquet file, chunk by chunk."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from sqlalchemy import text

        path = Path(path)
        writer = None
        try:
            with self._engine().connect() as con:
                for chunk in pd.read_sql(text(f"SELECT * FROM {self.from_sql()}"), con,
                                         chunksize=chunksize):
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema)
                    writer.write_table(table.cast(writer.schema))
        finally:
            if writer is not None:
                writer.close()
        return path

    # ── internals ───────────────────────────────────────────────────────────
    def _url(self):
        from sqlalchemy.engine import make_url
        return make_url(self.url)

    def _engine(self):
        from sqlalchemy import create_engine
        return create_engine(self.url)

    def _scalar(self, sql: str) -> Any:
        from sqlalchemy import text
        with self._engine().connect() as con:
            return con.execute(text(sql)).scalar()


def sql(url: str, table: Optional[str] = None, query: Optional[str] = None) -> SqlSource:
    """A database table or query as a FlashEDA source (needs SQLAlchemy)."""
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        raise ImportError('Database sources need SQLAlchemy: pip install "flasheda[sql]"') from None
    return SqlSource(url, table, query)
