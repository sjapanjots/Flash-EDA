from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union, Optional

import pandas as pd

from . import codegen, engine
from .engine import available_engines, choose_engine, inspect_source, sample, get_sample_size, set_sample_size
from .analyzers import overview, missing, numeric, categorical, correlation
from .report import EDAReport

__version__ = "0.2.0"
__all__ = [
    "analyze",
    "available_engines",
    "codegen",
    "engine",
    "get_sample_size",
    "inspect_source",
    "set_sample_size",
    "EDAReport",
]


def analyze(
    source: Union[pd.DataFrame, str, Path],
    n: Optional[int] = None,
    *,
    show: bool = False,
    random_state: int = 42,
    save_pdf: bool = True,
    pdf_path: str = "flasheda_report.pdf",
    engine: Optional[str] = None,
) -> EDAReport:
    """
    Analyse a fixed-size sample of `source` (DataFrame, CSV/TSV or Parquet path).

    `engine` forces the engine recommended for full-size work; by default it
    is chosen from the dataset's row count.
    """
    t0 = time.perf_counter()
    n = n or get_sample_size()

    info = inspect_source(source)
    sampled = sample(source, n=n, random_state=random_state)
    original_shape = info.shape

    tasks = {
        "overview":    lambda: overview.analyze(sampled, original_shape),
        "missing":     lambda: missing.analyze(sampled),
        "numeric":     lambda: numeric.analyze(sampled),
        "categorical": lambda: categorical.analyze(sampled),
        "correlation": lambda: correlation.analyze(sampled),
    }

    results: dict = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        future_map = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {"error": str(exc)}

    warnings: list[str] = []
    miss = results.get("missing", {})
    if miss.get("critical_columns"):
        warnings.append(f"Columns with >50% missing: {', '.join(miss['critical_columns'])}")
    num = results.get("numeric", {})
    if num.get("skewed_columns"):
        warnings.append(f"Highly skewed columns: {', '.join(num['skewed_columns'])}")
    cat = results.get("categorical", {})
    if cat.get("likely_id_columns"):
        warnings.append(f"Likely ID columns: {', '.join(cat['likely_id_columns'])}")
    corr = results.get("correlation", {})
    if corr.get("numeric_strong_pairs"):
        warnings.append(f"{len(corr['numeric_strong_pairs'])} strongly correlated pair(s) found")

    elapsed = round(time.perf_counter() - t0, 3)

    report = EDAReport(
        overview=results.get("overview", {}),
        missing=results.get("missing", {}),
        numeric=results.get("numeric", {}),
        categorical=results.get("categorical", {}),
        correlation=results.get("correlation", {}),
        sample_size=len(sampled),
        elapsed_seconds=elapsed,
        original_shape=original_shape,
        warnings=warnings,
        source=str(info.path) if info.path else None,
        recommended_engine=choose_engine(info.n_rows, prefer=engine),
        sample=sampled,
    )

    if show:
        report.show()

    if save_pdf:
        report.save_pdf(pdf_path)

    return report