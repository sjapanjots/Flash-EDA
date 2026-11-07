"""
flasheda.modeling.specs
~~~~~~~~~~~~~~~~~~~~~~~
Candidate models and evaluation settings, described as data so the
evaluator and the generated training script build exactly the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module: str
    cls: str
    params: Dict[str, Any] = field(default_factory=dict)
    impute: bool = True      # median-impute nulls first (HistGradientBoosting handles them)
    scale: bool = False      # standardise first (linear models)

    def build(self):
        import importlib

        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = getattr(importlib.import_module(self.module), self.cls)(**self.params)
        steps = []
        if self.impute:
            steps.append(SimpleImputer(strategy="median", keep_empty_features=True))
        if self.scale:
            steps.append(StandardScaler())
        return make_pipeline(*steps, model) if steps else model


RS = 42

CLASSIFIERS = (
    ModelSpec("baseline", "sklearn.dummy", "DummyClassifier", {"strategy": "prior"}, impute=False),
    ModelSpec("logistic_regression", "sklearn.linear_model", "LogisticRegression",
              {"max_iter": 1000}, scale=True),
    ModelSpec("random_forest", "sklearn.ensemble", "RandomForestClassifier",
              {"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": -1, "random_state": RS}),
    ModelSpec("hist_gradient_boosting", "sklearn.ensemble", "HistGradientBoostingClassifier",
              {"random_state": RS}, impute=False),
)

REGRESSORS = (
    ModelSpec("baseline", "sklearn.dummy", "DummyRegressor", {"strategy": "mean"}, impute=False),
    ModelSpec("ridge", "sklearn.linear_model", "Ridge", {}, scale=True),
    ModelSpec("random_forest", "sklearn.ensemble", "RandomForestRegressor",
              {"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": -1, "random_state": RS}),
    ModelSpec("hist_gradient_boosting", "sklearn.ensemble", "HistGradientBoostingRegressor",
              {"random_state": RS}, impute=False),
)


def candidates(task: str, random_state: int = RS) -> Tuple[ModelSpec, ...]:
    """Candidate models for `task`, seeded with `random_state`."""
    specs = REGRESSORS if task == "regression" else CLASSIFIERS
    return tuple(with_seed(s, random_state) for s in specs)


def with_seed(spec: ModelSpec, random_state: int) -> ModelSpec:
    """`spec` with its random_state parameter (if it has one) set to `random_state`."""
    if "random_state" not in spec.params:
        return spec
    return replace(spec, params={**spec.params, "random_state": random_state})


def scoring_for(target_info: Dict[str, Any]) -> Tuple[str, str]:
    """(scikit-learn scorer name, display label) matching the recommended metric."""
    task = target_info["task"]
    if task == "binary":
        if target_info.get("imbalanced"):
            return "average_precision", "PR-AUC"
        return "roc_auc", "ROC-AUC"
    if task == "multiclass":
        if target_info.get("imbalanced"):
            return "f1_macro", "macro-F1"
        return "accuracy", "accuracy"
    stats = target_info.get("target_stats", {})
    if stats.get("skewness", 0) > 1 and stats.get("min", -1) >= 0:
        return "neg_mean_absolute_error", "MAE"
    return "neg_root_mean_squared_error", "RMSE"
