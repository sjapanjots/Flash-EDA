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
from .specs import ModelSpec, candidates, scoring_for, with_seed

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
    warnings: List[str] = field(default_factory=list)
    holdout_rows: int = 0           # latest rows kept out of model selection (time-based only)
    beats_baseline: bool = True     # best model is clearly better than the baseline in CV

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
        has_holdout = "holdout" in self.leaderboard.columns
        cols = ["model", self.metric, "± std"] + (["holdout"] if has_holdout else []) + ["fit s"]
        for col in cols:
            table.add_column(col)
        for _, row in self.leaderboard.iterrows():
            style = "bold green" if row["model"] == self.best else None
            cells = [row["model"], f"{row['score']:.4f}", f"{row['std']:.4f}"]
            if has_holdout:
                cells.append("n/a" if pd.isna(row["holdout"]) else f"{row['holdout']:.4f}")
            table.add_row(*cells, f"{row['fit_seconds']:.2f}", style=style)
        console = Console()
        console.print(table)
        if has_holdout:
            console.print(f"  holdout = score on the latest {self.holdout_rows:,} rows, "
                          "never used to pick the model")
        for w in self.warnings:
            console.print(f"  [bold yellow]⚠  {w}[/bold yellow]")

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
    models: Optional[Sequence[Union[str, ModelSpec]]] = None,
    n_splits: int = 5,
    n: Optional[int] = None,
    random_state: int = 42,
    validate: bool = True,
    rare_classes: str = "error",
    holdout: float = 0.2,
    **plan_options,
) -> ModelResult:
    """
    Analyse `source`, build the cleaning + feature plan, compare models on
    the sample and return validated prep and training scripts.

    engine      : engine for the prep script (default: by size)
    output      : Parquet file the prep script writes and the training script reads
    models      : model names and/or your own ModelSpec objects (default: all
                  built-in candidates); the baseline is always compared too
    rare_classes: what to do with target classes too rare to evaluate (fewer
                  than 2 sampled rows): "error" (default), "drop" those rows,
                  or "merge" them into "other" (multiclass only)
    holdout     : with a time column, the share of the latest rows kept out of
                  model selection and scored once at the end (0 turns it off)
    random_state: seeds the sample, the CV shuffles and the models
    """
    from .. import analyze

    report = analyze(source, n=n, target=target, random_state=random_state, save_pdf=False)
    return train_from_report(report, engine, output=output, model_path=model_path,
                             models=models, n_splits=n_splits, validate=validate,
                             rare_classes=rare_classes, holdout=holdout,
                             random_state=random_state, **plan_options)


def train_from_report(
    report,
    engine: Optional[str] = None,
    *,
    output: str = "features.parquet",
    model_path: str = "model.joblib",
    models: Optional[Sequence[Union[str, ModelSpec]]] = None,
    n_splits: int = 5,
    validate: bool = True,
    rare_classes: str = "error",
    holdout: float = 0.2,
    random_state: int = 42,
    **plan_options,
) -> ModelResult:
    _require_sklearn()
    if not 0 <= holdout < 0.5:
        raise ValueError("holdout must be at least 0 and below 0.5")
    requested_splits = n_splits
    if rare_classes not in ("error", "drop", "merge"):
        raise ValueError("rare_classes must be 'error', 'drop' or 'merge'")
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
    notes: List[str] = []
    relabel: Dict[str, Any] = {}
    if task != "regression":
        data, relabel, notes = _handle_rare_classes(data, target, task, rare_classes)
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
    specs = _choose_models(task, models, random_state)

    # Time-based: the latest rows are an untouched holdout, scored once at the end.
    n_hold = int(len(X) * holdout) if time_order else 0
    if n_hold and n_hold < MIN_HOLDOUT_ROWS:
        notes.append(f"No holdout: {n_hold} rows would be too few to score (need {MIN_HOLDOUT_ROWS}).")
        n_hold = 0
    X_dev, y_dev = (X.iloc[:-n_hold], y.iloc[:-n_hold]) if n_hold else (X, y)
    cv, n_splits = _make_cv(task, y_dev, n_splits, bool(time_order), random_state)
    if task != "regression" and n_splits < requested_splits:
        notes.append(f"Cross-validation uses {n_splits} folds instead of {requested_splits}: "
                     "the rarest class has too few rows for more.")

    rows = []
    for spec in specs:
        row = _evaluate(spec, X_dev, y_dev, scoring, cv)
        if n_hold:
            row["holdout"] = _holdout_score(spec, X_dev, y_dev, X.iloc[-n_hold:], y.iloc[-n_hold:], scoring)
        rows.append(row)
    # scikit-learn scores are "higher is better" (error metrics are negated),
    # so sort descending, then flip error metrics back to positive for display.
    board = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    beats = _beats_baseline(board, n_splits)
    if not beats:
        notes.append("No model is clearly better than the baseline (a constant prediction) in "
                      "cross-validation: the features may carry little signal for this target. "
                      "Check the leakage and feature-power sections before trusting the model.")
    if scoring.startswith("neg_"):
        board["score"] = -board["score"]
        if n_hold:
            board["holdout"] = -board["holdout"]
    if n_hold and board["holdout"].isna().any():
        notes.append("Some holdout scores are n/a: the latest rows don't contain every target class.")
    best_row = board[board["model"] != "baseline"].iloc[0] if len(board) > 1 else board.iloc[0]
    best = next(s for s in specs if s.name == best_row["model"])

    summary = (f"best of {len(specs)} on a {len(X):,}-row sample "
               f"({label} {best_row['score']:.4f} ± {best_row['std']:.4f}"
               + (f", holdout {best_row['holdout']:.4f}" if n_hold and pd.notna(best_row["holdout"]) else "")
               + ")")
    code = render_training_script(
        best, task=task, target=target, features=features, scoring=scoring, metric_label=label,
        positive=positive, time_order=time_order, data=output, model_path=model_path,
        n_splits=n_splits, summary=summary, relabel=relabel,
        holdout=holdout if n_hold else 0.0, random_state=random_state,
    )
    if validate:
        _validate_training_script(code, data, features, target)

    return ModelResult(task, target, label, scoring, board, best.name, features, positive,
                       time_order, prep, code, report, notes, n_hold, beats)


def _choose_models(task: str, models, random_state: int) -> List[ModelSpec]:
    """Built-in names and custom ModelSpecs, seeded; the baseline is always included."""
    built_in = {s.name: s for s in candidates(task, random_state)}
    if models is None:
        return list(built_in.values())
    specs: List[ModelSpec] = []
    for m in models:
        if isinstance(m, ModelSpec):
            specs.append(with_seed(m, random_state))
        elif m in built_in:
            specs.append(built_in[m])
        else:
            raise ValueError(f"Unknown model {m!r}. Choose from {list(built_in)} or pass a ModelSpec.")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Model names must be unique: {names}")
    if "baseline" not in names:
        specs.insert(0, built_in["baseline"])
    return specs


def _beats_baseline(board: pd.DataFrame, n_splits: int) -> bool:
    """Best model's CV score above the baseline's by more than its standard error.
    Scores are still "higher is better" here (error metrics negated)."""
    if (board["model"] == "baseline").sum() != 1 or len(board) < 2:
        return True
    base = board.loc[board["model"] == "baseline", "score"].iloc[0]
    best = board[board["model"] != "baseline"].iloc[0]
    return bool(best["score"] - base > best["std"] / np.sqrt(max(n_splits, 1)))


MIN_HOLDOUT_ROWS = 30       # fewer holdout rows give a meaningless score
MIN_CLASS_ROWS = 2          # a class needs rows in both the training and test part of a fold
UNSTABLE_CLASS_ROWS = 20    # below this, scores for the class swing a lot between samples


def _handle_rare_classes(data: pd.DataFrame, target: str, task: str, policy: str):
    """Classes with fewer than MIN_CLASS_ROWS sampled rows can't be evaluated."""
    labels = data[target].astype(str)
    counts = labels.value_counts()
    rare = counts[counts < MIN_CLASS_ROWS]
    notes: List[str] = []
    relabel: Dict[str, Any] = {}
    if len(rare):
        listing = ", ".join(f"{v!r} ({c} row{'s' if c != 1 else ''})" for v, c in rare.items())
        if task == "binary" or policy == "error":
            raise ValueError(
                f"Target class(es) {listing} have fewer than {MIN_CLASS_ROWS} rows in the "
                f"{len(data):,}-row sample, too few to train and evaluate a model. "
                "Use a larger sample (fe.train(..., n=50_000)) or more data"
                + ("; for multiclass targets, rare_classes='drop' or 'merge' also works." if task == "multiclass" else "."))
        if policy == "drop":
            data = data[~labels.isin(rare.index)]
            relabel = {"drop": sorted(str(v) for v in rare.index)}
            notes.append(f"Dropped rows of rare target classes: {listing}.")
        else:
            other = "other"
            while other in counts.index:
                other = f"_{other}_"
            data = data.assign(**{target: labels.where(~labels.isin(rare.index), other)})
            relabel = {"merge": sorted(str(v) for v in rare.index), "into": other}
            notes.append(f"Merged rare target classes into {other!r}: {listing}.")
            if (data[target] == other).sum() < MIN_CLASS_ROWS:
                raise ValueError(f"Even merged, the rare classes have fewer than {MIN_CLASS_ROWS} rows.")
    smallest = data[target].astype(str).value_counts()
    if smallest.min() < UNSTABLE_CLASS_ROWS:
        notes.append(f"Class {smallest.idxmin()!r} has only {smallest.min()} sampled rows: "
                     "its scores will vary a lot between samples. Consider a larger sample (n=...).")
    return data, relabel, notes


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


def _holdout_score(spec: ModelSpec, X_dev, y_dev, X_hold, y_hold, scoring: str) -> float:
    """Fit on the earlier rows, score once on the latest rows (NaN if the scorer can't)."""
    from sklearn.metrics import get_scorer

    try:
        return float(get_scorer(scoring)(spec.build().fit(X_dev, y_dev), X_hold, y_hold))
    except ValueError:          # e.g. ROC-AUC when the holdout has a single class
        return float("nan")


def _make_cv(task: str, y: pd.Series, n_splits: int, time_based: bool, random_state: int = 42):
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

    if time_based:
        if task == "regression":
            return TimeSeriesSplit(n_splits=n_splits), n_splits
        for k in range(n_splits, 1, -1):
            folds = TimeSeriesSplit(n_splits=k).split(np.zeros(len(y)))
            if all(y.iloc[tr].nunique() == y.iloc[te].nunique() == y.nunique() for tr, te in folds):
                return TimeSeriesSplit(n_splits=k), k
        raise ValueError(
            "A time-based split can't put every target class in both the training and the test "
            "part of each fold: the rare class is too rare or only occurs in one period. "
            "Use a larger sample (n=...) or more data.")
    if task == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_splits
    smallest = int(y.value_counts().min())
    k = max(2, min(n_splits, smallest))
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state), k


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
