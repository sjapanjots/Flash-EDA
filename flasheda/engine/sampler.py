"""
flasheda.engine.sampler
~~~~~~~~~~~~~~~~~~~~~~~
Uniform random sampling of exactly `n` rows (or all rows, if fewer)
from any source, without ever loading the full dataset into memory.

Strategy, fastest first:
  1. DuckDB native reservoir sampling — C++ scan, out-of-core
  2. Vectorised "random key" sampling over chunks — every row gets a
     uniform random key and the `n` smallest keys are kept. This is an
     exact uniform sample, computed with numpy instead of a Python loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import pandas as pd

from .backends import is_available
from .source import CSV_EXTS, DUCKDB_CSV_TYPES, PARQUET_EXTS, _is_polars_frame, csv_sep

DEFAULT_SAMPLE_SIZE = 5_000
_sample_size = DEFAULT_SAMPLE_SIZE
CHUNK_ROWS = 250_000


def get_sample_size() -> int:
    return _sample_size


def set_sample_size(n: int) -> None:
    global _sample_size
    if n < 100:
        raise ValueError("Sample size must be at least 100.")
    _sample_size = n


def sample(
    source: Any,
    n: Optional[int] = None,
    random_state: int = 42,
    method: str = "auto",
) -> pd.DataFrame:
    """
    Return a uniform random sample of `n` rows as a pandas DataFrame.

    method: "auto" (DuckDB if installed), "duckdb" or "chunked".
    """
    n = n or get_sample_size()

    if isinstance(source, pd.DataFrame):
        if len(source) <= n:
            return source.reset_index(drop=True)
        return source.sample(n=n, random_state=random_state).reset_index(drop=True)

    if _is_polars_frame(source):
        out = source if source.height <= n else source.sample(n=n, seed=random_state)
        return out.to_pandas()

    if not isinstance(source, (str, Path)):
        raise TypeError(f"source must be a DataFrame or file path, got {type(source)}")

    path = Path(source)
    ext = path.suffix.lower()
    if ext not in CSV_EXTS + PARQUET_EXTS:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv, .tsv or .parquet")

    if method == "duckdb" or (method == "auto" and is_available("duckdb")):
        df = _sample_duckdb(path, n, random_state)
    elif method in ("auto", "chunked"):
        df = _sample_chunked(_iter_chunks(path), n, random_state)
    else:
        raise ValueError(f"Unknown sampling method {method!r}")

    if df.empty:
        raise ValueError("File appears to be empty.")
    return df


def _sample_duckdb(path: Path, n: int, random_state: int) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect()
    # A seeded reservoir sample is only reproducible on a single thread.
    con.execute("SET threads = 1")
    if path.suffix.lower() in PARQUET_EXTS:
        scan = "read_parquet(?)"
        params = [str(path)]
    else:
        scan = f"read_csv(?, delim = ?, header = true, {DUCKDB_CSV_TYPES})"
        params = [str(path), csv_sep(path)]
    query = (
        f"SELECT * FROM {scan} "
        f"USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE ({int(random_state)})"
    )
    return con.execute(query, params).df().reset_index(drop=True)


def _iter_chunks(path: Path) -> Iterator[pd.DataFrame]:
    if path.suffix.lower() in PARQUET_EXTS:
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=CHUNK_ROWS):
            yield batch.to_pandas()
    else:
        yield from pd.read_csv(path, sep=csv_sep(path), chunksize=CHUNK_ROWS, low_memory=False)


def _sample_chunked(chunks: Iterator[pd.DataFrame], n: int, random_state: int) -> pd.DataFrame:
    """Keep the n rows with the smallest uniform random keys across all chunks."""
    rng = np.random.default_rng(random_state)
    kept: Optional[pd.DataFrame] = None
    key = "__flasheda_key__"

    for chunk in chunks:
        chunk = chunk.assign(**{key: rng.random(len(chunk))})
        kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        if len(kept) > n:
            kept = kept.nsmallest(n, key)

    if kept is None:
        return pd.DataFrame()
    return kept.drop(columns=key).reset_index(drop=True)
