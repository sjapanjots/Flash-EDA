import runpy

import joblib
import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.codegen import DateParts, FrequencyEncode, OneHot, Ratio, compare_frames

RUNNABLE = ["pandas", "polars", "duckdb"]
FAST = ["baseline", "logistic_regression"]       # keeps the suite quick


def make_shop(n: int = 4_000, seed: int = 5, task: str = "binary") -> pd.DataFrame:
    """Orders where the target depends on spend / visits — a ratio feature."""
    rng = np.random.default_rng(seed)
    visits = rng.integers(1, 50, n).astype(float)
    spend = visits * rng.lognormal(2, 0.4, n)
    per_visit = spend / visits
    when = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 10**6, n), unit="min")
    df = pd.DataFrame({
        "order_time": when.strftime("%Y-%m-%d %H:%M:%S"),
        "spend": spend.round(2),
        "visits": visits,
        "channel": rng.choice(["web", "app", "store"], n),
        "region": rng.choice([f"r{i}" for i in range(40)], n),
        "note": [f"free text {i} {rng.integers(10**6)}" for i in range(n)],
    })
    if task == "binary":
        df["vip"] = np.where(per_visit > np.quantile(per_visit, 0.85), "yes", "no")   # 15%: imbalanced
    elif task == "multiclass":
        df["tier"] = pd.qcut(per_visit, 3, labels=["low", "mid", "high"]).astype(str)   # balanced
    else:
        df["value"] = per_visit * 10 + rng.normal(0, 1, n)
    return df


@pytest.fixture(scope="module")
def shop():
    return make_shop()


@pytest.fixture(scope="module")
def report(shop):
    return fe.analyze(shop, target="vip", save_pdf=False)


# ── features ─────────────────────────────────────────────────────────────────

def test_feature_plan(report):
    steps = report.plan(features=True).steps
    kinds = {type(s) for s in steps}
    assert {DateParts, OneHot, FrequencyEncode, Ratio} <= kinds
    dates = next(s for s in steps if isinstance(s, DateParts))
    assert dates.column == "order_time" and "hour" in dates.parts
    assert next(s for s in steps if isinstance(s, OneHot)).column == "channel"
    assert next(s for s in steps if isinstance(s, FrequencyEncode)).column == "region"
    ratio = next(s for s in steps if isinstance(s, Ratio))
    assert {ratio.numerator, ratio.denominator} == {"spend", "visits"}
    text = "\n".join(report.plan(features=True).describe())
    assert "note: 4000 distinct values" in text or "note:" in text      # free text dropped


def test_features_off_by_default(report):
    assert not any(isinstance(s, (DateParts, OneHot, Ratio)) for s in report.plan().steps)


def test_feature_options(report):
    steps = report.plan(features={"ratios": False, "dates": False}).steps
    assert not any(isinstance(s, (Ratio, DateParts)) for s in steps)


@pytest.mark.parametrize("eng", RUNNABLE)
def test_feature_code_matches_reference(report, eng):
    result = report.codegen(eng, features=True, validate=False).validate()
    assert result.ok and not result.skipped, str(result)


def test_featured_data_is_model_ready(report):
    out = report.plan(features=True).apply(report.sample)
    non_numeric = [c for c in out.columns if c != "vip"
                   and not (pd.api.types.is_numeric_dtype(out[c]) or pd.api.types.is_bool_dtype(out[c]))]
    assert non_numeric == []


# ── modeling ─────────────────────────────────────────────────────────────────

def test_train_binary(report):
    res = report.train(models=FAST)
    assert res.task == "binary" and res.metric == "PR-AUC" and res.positive_label == "yes"
    board = res.leaderboard.set_index("model")["score"]
    assert board["logistic_regression"] > board["baseline"] + 0.2
    assert res.best == "logistic_regression"
    assert res.time_order[:1] == ["order_time_year"]          # date column → time-based split
    assert "TimeSeriesSplit" in res.train_code


@pytest.mark.parametrize("task,target,metric,models", [
    ("multiclass", "tier", "accuracy", FAST),
    ("regression", "value", "MAE", ["baseline", "ridge"]),      # right-skewed target
])
def test_train_other_tasks(task, target, metric, models):
    res = fe.train(make_shop(2_000, task=task), target=target, models=models)
    assert res.task == task and res.metric == metric
    assert res.leaderboard["model"].tolist()[0] != "baseline"


def test_error_metrics_rank_lowest_error_first():
    res = fe.train(make_shop(2_000, task="regression"), target="value", models=["baseline", "ridge"])
    assert res.lower_is_better
    assert res.leaderboard["score"].is_monotonic_increasing
    assert res.best == "ridge"


def test_unknown_model_and_missing_target(report, shop):
    with pytest.raises(ValueError, match="No models match"):
        report.train(models=["xgboost"])
    with pytest.raises(ValueError, match="needs a target"):
        fe.analyze(shop, save_pdf=False).train()


@pytest.mark.parametrize("eng", ["polars", "duckdb"])
def test_prep_and_train_scripts_end_to_end(shop, tmp_path, eng):
    src = tmp_path / "shop.csv"
    shop.to_csv(src, index=False)
    data, model = tmp_path / "features.parquet", tmp_path / "model.joblib"
    res = fe.train(src, target="vip", engine=eng, output=str(data), model_path=str(model),
                   models=FAST)
    paths = res.save(tmp_path / "prep.py", tmp_path / "train.py")

    runpy.run_path(str(paths["prep"]), run_name="__main__")
    assert compare_frames(res.prep.plan.apply(pd.read_csv(src)), pd.read_parquet(data)) is None

    runpy.run_path(str(paths["train"]), run_name="__main__")
    fitted = joblib.load(model)
    X = pd.read_parquet(data)[res.features]
    assert fitted.predict_proba(X).shape == (len(X), 2)
