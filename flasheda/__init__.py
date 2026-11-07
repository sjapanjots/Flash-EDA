from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Union, Optional

import pandas as pd

from . import codegen, engine
from .clean import clean
from .config import CleanConfig
from .modeling import train
from .contract import check
from .compare import compare
from .timeseries import timeseries
from .engine import available_engines, choose_engine, inspect_source, sample, sample_size, get_sample_size, set_sample_size, sql
from .analyzers import overview, missing, numeric, categorical, correlation
from .analyzers import target as target_analyzer
from .analyzers import text as text_analyzer
from .report import EDAReport

__version__ = "0.6.0"
__all__ = [
    "CleanConfig",
    "analyze",
    "available_engines",
    "check",
    "clean",
    "compare",
    "codegen",
    "engine",
    "get_sample_size",
    "inspect_source",
    "sample_size",
    "set_sample_size",
    "sql",
    "timeseries",
    "train",
    "EDAReport",
]


def analyze(
    source: Union[pd.DataFrame, str, Path],
    n: Optional[int] = None,
    *,
    show: bool = False,
    random_state: int = 42,
    save_pdf: bool = False,
    pdf_path: str = "flasheda_report.pdf",
    engine: Optional[str] = None,
    target: Optional[str] = None,
    strict: bool = False,
) -> EDAReport:
    """
    Analyse a fixed-size sample of `source`: a DataFrame, a CSV/TSV or Parquet
    path, or a database table/query from flasheda.sql().

    Nothing is written to disk unless you ask: pass save_pdf=True (to
    pdf_path) or call report.save_pdf() / report.save_html().

    Each section is computed independently. If one fails, the rest of the
    report is still produced, the failure is listed in `report.errors` and in
    `report.warnings`; pass strict=True to raise instead.

    `target` names the column you want to predict; it adds task detection,
    class balance, per-feature predictive power and leakage checks.

    `engine` forces the engine recommended for full-size work; by default it
    is chosen from the dataset's row count.
    """
    t0 = time.perf_counter()
    n = n or get_sample_size()

    info = inspect_source(source)
    if target is not None and target not in info.columns:
        raise KeyError(f"Target column {target!r} not found. Columns: {info.columns}")
    sampled = sample(source, n=n, random_state=random_state)
    original_shape = info.shape

    tasks = {
        "overview":    lambda: overview.analyze(sampled, original_shape),
        "missing":     lambda: missing.analyze(sampled),
        "numeric":     lambda: numeric.analyze(sampled),
        "categorical": lambda: categorical.analyze(sampled),
        "correlation": lambda: correlation.analyze(sampled),
        "text":        lambda: text_analyzer.analyze(sampled),
    }
    if target is not None:
        tasks["target"] = lambda: target_analyzer.analyze(sampled, target, random_state)

    results: dict = {}
    errors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        future_map = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                if strict or name == "target":
                    raise RuntimeError(f"The {name} analysis failed: {type(exc).__name__}: {exc}") from exc
                errors[name] = f"{type(exc).__name__}: {exc}"
                results[name] = {}

    warnings: list[str] = [
        f"The {name} analysis failed and is missing from this report ({msg})"
        for name, msg in sorted(errors.items())
    ]
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
    for p in corr.get("numeric_boolean_strong_pairs", []):
        warnings.append(f"{p['numeric']!r} almost determines {p['boolean']!r} "
                        f"(point-biserial r = {p['point_biserial_r']})")

    txt = results.get("text", {})
    for col in txt.get("personal_data_columns", []):
        kinds = ", ".join(f"{k} {v}%" for k, v in txt["columns"][col]["personal_data"].items())
        warnings.append(f"Possible personal data in {col!r}: {kinds} of non-empty values")

    tgt = results.get("target", {})
    if tgt.get("leakage_columns"):
        cols = ", ".join(str(l["column"]) for l in tgt["leakage_columns"])
        warnings.append(f"Possible target leakage: {cols}")
    if tgt.get("imbalanced"):
        warnings.append(
            f"Imbalanced target: minority class is {tgt['minority_pct']}% "
            f"(ratio {tgt['imbalance_ratio']}:1)"
        )
    if tgt.get("null_count"):
        warnings.append(f"Target {target!r} is null in {tgt['null_pct']}% of sampled rows")

    elapsed = round(time.perf_counter() - t0, 3)

    report = EDAReport(
        overview=results.get("overview", {}),
        missing=results.get("missing", {}),
        numeric=results.get("numeric", {}),
        categorical=results.get("categorical", {}),
        correlation=results.get("correlation", {}),
        text=txt,
        sample_size=len(sampled),
        elapsed_seconds=elapsed,
        original_shape=original_shape,
        warnings=warnings,
        target=tgt,
        errors=errors,
        source=str(info.path) if info.path else None,
        sql_source=info.sql,
        # database sources are read with pandas (or exported with .to_parquet())
        recommended_engine=choose_engine(
            info.n_rows, prefer=engine or ("pandas" if info.sql is not None else None)),
        sample=sampled,
    )

    if show:
        report.show()

    if save_pdf:
        report.save_pdf(pdf_path)

    return report