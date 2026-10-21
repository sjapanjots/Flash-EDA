"""
flasheda.clean
~~~~~~~~~~~~~~
One call from raw data to a validated cleaning script:

    code = flasheda.clean("orders.csv", target="churn", engine="polars")
    code.save("clean_orders.py")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from . import detectors

__all__ = ["clean", "detectors"]


def clean(
    source: Any,
    target: Optional[str] = None,
    engine: Optional[str] = None,
    *,
    output: str = "cleaned.parquet",
    n: Optional[int] = None,
    random_state: int = 42,
    validate: bool = True,
    features: bool = False,
    **plan_options,
):
    """
    Analyse `source`, plan the cleaning and return the generated script
    (a `GeneratedCode`, already validated on the sample).

    target       : column to predict; enables leakage drops, null-target
                   row drops and missing-value flags
    engine       : "pandas" | "polars" | "duckdb" | "pyspark" (default: by size)
    features     : also add model-ready features (date parts, encodings, ratios)
    plan_options : switch planner phases off, e.g. group_rare=False
    """
    from .. import analyze  # local import: flasheda/__init__ imports this package

    report = analyze(source, n=n, target=target, random_state=random_state, save_pdf=False)
    code = report.codegen(engine, output=output, validate=validate, features=features,
                          **plan_options)
    code.report = report
    return code
