"""
flasheda.modeling
~~~~~~~~~~~~~~~~~
Quick, honest baselines on the sample, then scripts for the full data:

    result = flasheda.train("orders.csv", target="churn", engine="polars")
    result.show()                      # leaderboard
    result.save("prep.py", "train.py")
    #  python prep.py   → cleaned + featured Parquet (any engine, full data)
    #  python train.py  → cross-validated score + model.joblib

Models are compared with the metric and split recommended by the target
analysis (time-based when there's a date column). Requires scikit-learn:
pip install "flasheda[ml]".
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .script import render_training_script
from .specs import ModelSpec, candidates, scoring_for

__all__ = ["ModelResult", "ModelSpec", "train", "train_from_report"]


@dataclass
class ModelResult:
    task: str
    target: str
    metric: str                     # display label, e.g. "PR-AUC"
    scoring: str                    # scikit-learn scorer name
    leaderboard: pd.DataFrame
    best: str
    features: List[str]
    positive_label: Optional[str]
    time_order: List[str]
    prep: Any                       # GeneratedCode: cleaning + features
    train_code: str
    report: Any = field(default=None, repr=False)

    @property
    def lower_is_better(self) -> bool:
        return self.scoring.startswith("neg_")

    def save(self, prep_path: Union[str, Path] = "prep.py",
             train_path: Union[str, Path] = "train.py") -> Dict[str, Path]:
        return {"prep": self.prep.save(prep_path),
                "train": _write(train_path, self.train_code)}

    def show(self) -> None:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"{self.task} · {self.target} · {self.metric} "
                            f"({'lower' if self.lower_is_better else 'higher'} is better)",
                      box=box.SIMPLE)
        for col in ("model", self.metric, "± std", "fit s"):
            table.add_column(col)
        for _, row in self.leaderboard.iterrows():
            style = "bold green" if row["model"] == self.best else None
            table.add_row(row["model"], f"{row['score']:.4f}", f"{row['std']:.4f}",
                          f"{row['fit_seconds']:.2f}", style=style)
        Console().print(table)

    def __repr__(self) -> str:
        top = self.leaderboard.iloc[0]
        return (f"ModelResult(task={self.task!r}, best={self.best!r}, "
                f"{self.metric}={top['score']:.4f}, features={len(self.features)})")


def train(
    source: Any,
    target: str,
    engine: Optional[str] = None,
    *,
    output: str = "features.parquet",
    model_path: str = "model.joblib",
    models: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    n: Optional[int] = None,
    random_state: int = 42,
    validate: bool = True,
    **plan_options,
) -> ModelResult:
    """
    Analyse `source`, build the cleaning + feature plan, compare models on
    the sample and return validated prep and training scripts.

    engine      : engine for the prep script (default: by size)
    output      : Parquet file the prep script writes and the training script reads
    models      : subset of model names (default: all candidates for the task)
    """
    from .. import analyze

    report = analyze(source, n=n, target=target, random_state=random_state, save_pdf=False)
    return train_from_report(report, engine, output=output, model_path=model_path,
                             models=models, n_splits=n_splits, validate=validate, **plan_options)


def train_from_report(
    report,
    engine: Optional[str] = None,
    *,
    output: str = "features.parquet",
    model_path: str = "model.joblib",
    models: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    validate: bool = True,
    **plan_options,
) -> ModelResult:
    _require_sklearn()
    from ..features import time_order_columns

    t = report.target
    if not t:
        raise ValueError("Modeling needs a target: run flasheda.analyze(source, target='column').")
    target, task = t["target"], t["task"]

    prep = report.codegen(engine, output=output, validate=validate, features=True, **plan_options)
    data = prep.plan.apply(report.sample)
    time_order = time_order_columns(prep.plan, t.get("time_columns", []))

    features = [c for c in data.columns if c != target and (
        pd.api.types.is_numeric_dtype(data[c]) or pd.api.types.is_bool_dtype(data[c]))]
    data = data[data[target].notna()]
    if time_order:
        data = data.sort_values(time_order, na_position="first", kind="stable")
    X = data[features]
    positive = None
    if task == "binary":
        positive = t["classes"][-1]["value"]           # minority class
        y = (data[target].astype(str) == positive).astype(int)
    elif task == "multiclass":
        y = data[target].astype(str)
    else:
        y = data[target].astype(float)

    scoring, label = scoring_for(t)
    specs = [s for s in candidates(task) if models is None or s.name in models]
    if not specs:
        raise ValueError(f"No models match {models!r}. Choose from {[s.name for s in candidates(task)]}.")
    cv, n_splits = _make_cv(task, y, n_splits, bool(time_order))

    rows = []
    for spec in specs:
        rows.append(_evaluate(spec, X, y, scoring, cv))
    # scikit-learn scores are "higher is better" (error metrics are negated),
    # so sort descending, then flip error metrics back to positive for display.
    board = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if scoring.startswith("neg_"):
        board["score"] = -board["score"]
    best_row = board[board["model"] != "baseline"].iloc[0] if len(board) > 1 else board.iloc[0]
    best = next(s for s in specs if s.name == best_row["model"])

    summary = (f"best of {len(specs)} on a {len(X):,}-row sample "
               f"({label} {best_row['score']:.4f} ± {best_row['std']:.4f})")
    code = render_training_script(
        best, task=task, target=target, features=features, scoring=scoring, metric_label=label,
        positive=positive, time_order=time_order, data=output, model_path=model_path,
        n_splits=n_splits, summary=summary,
    )
    if validate:
        _validate_training_script(code, data, features, target)

    return ModelResult(task, target, label, scoring, board, best.name, features, positive,
                       time_order, prep, code, report)


def _evaluate(spec: ModelSpec, X, y, scoring: str, cv) -> Dict[str, Any]:
    from sklearn.model_selection import cross_validate

    t0 = time.perf_counter()
    res = cross_validate(spec.build(), X, y, scoring=scoring, cv=cv, error_score="raise")
    return {
        "model": spec.name,
        "score": float(np.mean(res["test_score"])),
        "std": float(np.std(res["test_score"])),
        "fit_seconds": round(time.perf_counter() - t0, 2),
    }


def _make_cv(task: str, y: pd.Series, n_splits: int, time_based: bool):
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

    if time_based:
        return TimeSeriesSplit(n_splits=n_splits), n_splits
    if task == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=42), n_splits
    smallest = int(y.value_counts().min())
    k = max(2, min(n_splits, smallest))
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42), k


def _validate_training_script(code: str, data: pd.DataFrame, features, target) -> None:
    """Run the generated script on the sample (2 folds) so users never get broken code."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.parquet")
        data.to_parquet(path, index=False)
        ns: Dict[str, Any] = {"__name__": "flasheda_generated_train"}
        exec(compile(code, "<flasheda:train>", "exec"), ns)
        ns.update(DATA=path, MODEL_PATH=os.path.join(tmp, "model.joblib"), N_SPLITS=2)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ns["main"]()
        if not os.path.exists(ns["MODEL_PATH"]):
            raise RuntimeError("Generated training script did not save a model.")


def _write(path: Union[str, Path], text: str) -> Path:
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return p


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError('Modeling needs scikit-learn: pip install "flasheda[ml]"') from None
