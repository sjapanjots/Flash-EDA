from .base import Writer
from .duckdb_writer import DuckDBWriter
from .pandas_writer import PandasWriter
from .polars_writer import PolarsWriter
from .pyspark_writer import PySparkWriter

WRITERS = {
    "pandas": PandasWriter,
    "polars": PolarsWriter,
    "duckdb": DuckDBWriter,
    "pyspark": PySparkWriter,
}


def get_writer(engine: str) -> Writer:
    try:
        return WRITERS[engine]()
    except KeyError:
        raise ValueError(f"No code writer for engine {engine!r}. Choose from {list(WRITERS)}.")


__all__ = ["WRITERS", "Writer", "get_writer"]
