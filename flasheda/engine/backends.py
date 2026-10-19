"""
flasheda.engine.backends
~~~~~~~~~~~~~~~~~~~~~~~~
Detects which dataframe engines are installed and picks the right one
for a dataset of a given size.

  pandas  — always available, best for data that fits comfortably in RAM
  polars  — lazy, multi-threaded, streams larger-than-memory files
  duckdb  — out-of-core SQL, great for files and databases
  pyspark — distributed, for cluster-scale data
"""

from __future__ import annotations

import importlib.util
from typing import Dict, Optional

ENGINES = ("pandas", "polars", "duckdb", "pyspark")

# Row-count thresholds for automatic engine selection
SMALL_ROWS = 1_000_000        # below this pandas is simplest and fast enough
LARGE_ROWS = 200_000_000      # above this a distributed engine pays off


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_available(engine: str) -> bool:
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}. Choose from {ENGINES}.")
    return has_module(engine)


def available_engines() -> Dict[str, bool]:
    return {name: is_available(name) for name in ENGINES}


def choose_engine(n_rows: Optional[int], prefer: Optional[str] = None) -> str:
    """
    Pick an engine for a dataset with `n_rows` rows.

    `prefer` forces a specific engine (it must be installed).
    Unknown row counts are treated as large.
    """
    if prefer:
        if not is_available(prefer):
            raise ImportError(
                f"Engine {prefer!r} is not installed. "
                f"Install it with: pip install flasheda[{_extra_for(prefer)}]"
            )
        return prefer

    rows = n_rows if n_rows is not None and n_rows >= 0 else LARGE_ROWS

    if rows < SMALL_ROWS:
        return "pandas"
    if rows >= LARGE_ROWS and is_available("pyspark"):
        return "pyspark"
    for engine in ("polars", "duckdb"):
        if is_available(engine):
            return engine
    return "pandas"


def _extra_for(engine: str) -> str:
    return {"polars": "fast", "duckdb": "fast", "pyspark": "spark"}.get(engine, "fast")
