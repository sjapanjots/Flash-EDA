"""
flasheda.codegen.validate
~~~~~~~~~~~~~~~~~~~~~~~~~
Runs generated code on the sample and checks it matches the reference
`Plan.apply()` result, so users never receive broken code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..engine.backends import is_available
from .steps import Plan


@dataclass
class ValidationResult:
    engine: str
    ok: bool
    skipped: bool = False
    message: str = ""

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        status = "SKIPPED" if self.skipped else ("OK" if self.ok else "FAILED")
        return f"[{status}] {self.engine}: {self.message}"


def run_generated(code: str, engine: str, df: pd.DataFrame) -> pd.DataFrame:
    """Execute generated code's transform() on `df`, returning pandas."""
    namespace: dict = {"__name__": "flasheda_generated"}
    exec(compile(code, f"<flasheda:{engine}>", "exec"), namespace)
    transform = namespace["transform"]

    if engine == "pandas":
        return transform(df)
    if engine == "polars":
        import polars as pl
        return transform(pl.from_pandas(df).lazy()).collect().to_pandas()
    if engine == "duckdb":
        import duckdb
        con = duckdb.connect()
        con.register("source", df)
        return transform(con, "source").df()
    if engine == "pyspark":
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        return transform(spark.createDataFrame(df)).toPandas()
    raise ValueError(f"Unknown engine {engine!r}")


def validate_code(code: str, engine: str, plan: Plan, sample: pd.DataFrame) -> ValidationResult:
    if not is_available(engine):
        return ValidationResult(engine, ok=True, skipped=True, message=f"{engine} not installed")
    try:
        expected = plan.apply(sample)
        actual = run_generated(code, engine, sample)
    except Exception as exc:  # report, don't raise — the caller decides
        return ValidationResult(engine, ok=False, message=f"{type(exc).__name__}: {exc}")

    problem = compare_frames(expected, actual)
    if problem:
        return ValidationResult(engine, ok=False, message=problem)
    return ValidationResult(
        engine, ok=True, message=f"matches reference on {len(sample):,} sampled rows"
    )


def compare_frames(expected: pd.DataFrame, actual: pd.DataFrame) -> Optional[str]:
    """Order-insensitive comparison; numbers compared to 6 significant digits."""
    exp_cols = [str(c) for c in expected.columns]
    act_cols = [str(c) for c in actual.columns]
    if sorted(exp_cols) != sorted(act_cols):
        return f"column mismatch: expected {exp_cols}, got {act_cols}"
    if len(expected) != len(actual):
        return f"row count mismatch: expected {len(expected)}, got {len(actual)}"

    a = _canonical(expected, exp_cols)
    b = _canonical(actual.set_axis(act_cols, axis=1), exp_cols)
    if a != b:
        diff = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        return f"values differ, e.g. expected {a[diff]} got {b[diff]}"
    return None


def _canonical(df: pd.DataFrame, cols: list) -> list:
    rows = []
    for record in df[cols].itertuples(index=False, name=None):
        rows.append(tuple(_cell(v) for v in record))
    return sorted(rows)


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and v != v) or v is pd.NA or v is pd.NaT:
        return "<null>"
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)
