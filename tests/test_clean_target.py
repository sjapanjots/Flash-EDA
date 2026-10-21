import runpy

import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.analyzers import target as target_analyzer
from flasheda.clean import detectors
from flasheda.codegen import compare_frames

RUNNABLE = ["pandas", "polars", "duckdb"]


def make_churn(n: int = 8_000, seed: int = 3) -> pd.DataFrame:
    """Churn data with every problem the target-aware cleaning handles."""
    rng = np.random.default_rng(seed)
    churn = rng.random(n) < 0.12
    income = rng.lognormal(10, 0.8, n)
    income_missing = rng.random(n) < np.where(churn, 0.45, 0.05)   # missingness predicts churn
    price = rng.lognormal(3, 0.6, n).round(2).astype(str).astype(object)
    price[rng.random(n) < 0.03] = "N/A"
    city = rng.choice(["Delhi", "delhi ", "DELHI", "Pune", "pune", "Mumbai"], n).astype(object)
    source = rng.choice([f"ref_{i}" for i in range(60)], n, p=_tail(60, rng)).astype(object)
    label = np.where(churn, "yes", "no").astype(object)
    label[rng.random(n) < 0.02] = None
    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "income": np.where(income_missing, np.nan, income),
        "price": price,
        "city": city,
        "source": source,
        "refund_issued": np.where(rng.random(n) < 0.99, churn, ~churn),     # leak
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="h").astype(str),
        "churn": label,
    })
    return pd.concat([df, df.iloc[:100].assign(customer_id=lambda d: d.customer_id + n)],
                     ignore_index=True)


def _tail(k, rng):
    w = 1 / np.arange(1, k + 1) ** 1.6
    return w / w.sum()


@pytest.fixture(scope="module")
def churn():
    return make_churn()


@pytest.fixture(scope="module")
def report(churn):
    return fe.analyze(churn, target="churn", save_pdf=False)


# ── target analysis ──────────────────────────────────────────────────────────

def test_target_summary(report):
    t = report.target
    assert t["task"] == "binary"
    assert t["imbalanced"] and t["minority_pct"] < 20
    assert t["null_count"] > 0
    assert "PR-AUC" in t["recommended_metric"]
    assert t["time_columns"] == ["signup_date"]
    assert "time-based" in t["recommended_split"]


def test_leak_flagged_and_ids_score_zero(report):
    power = {p["column"]: p["score"] for p in report.target["feature_power"]}
    assert [l["column"] for l in report.target["leakage_columns"]] == ["refund_issued"]
    assert power["customer_id"] < 0.01
    assert power["refund_issued"] > 0.8
    assert any("leakage" in w for w in report.warnings)


@pytest.mark.parametrize("task,y", [
    ("binary", pd.Series([True, False] * 50)),
    ("multiclass", pd.Series(list("abc") * 40)),
    ("multiclass", pd.Series([1, 2, 3, 4] * 30)),
    ("regression", pd.Series(np.linspace(0, 1, 200))),
])
def test_detect_task(task, y):
    assert target_analyzer.detect_task(y) == task


def test_regression_target_finds_linear_leak():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.lognormal(3, 1, 3_000), "noise": rng.random(3_000)})
    df["y_times_two"] = df.y * 2 + rng.normal(0, 0.1, 3_000)
    t = target_analyzer.analyze(df, "y")
    assert t["task"] == "regression"
    assert [l["column"] for l in t["leakage_columns"]] == ["y_times_two"]
    assert "RMSLE" in t["recommended_metric"]


def test_unknown_target_raises(churn):
    with pytest.raises(KeyError, match="nope"):
        fe.analyze(churn, target="nope", save_pdf=False)


# ── detectors ────────────────────────────────────────────────────────────────

def test_detectors():
    assert detectors.is_numeric_text(pd.Series(["1.5", " 2 ", "N/A"] + ["3"] * 60))
    assert not detectors.is_numeric_text(pd.Series(["01234", "00981"] * 30))   # codes, not numbers
    assert detectors.text_variants(pd.Series(["Pune", "pune ", "Goa"])) == (3, 2)
    assert detectors.text_variants(pd.Series(["Pune", "Goa"])) is None
    assert detectors.is_numeric_id(pd.Series(range(100)), "user_id")
    assert not detectors.is_numeric_id(pd.Series(range(100)), "amount")
    rare = detectors.rare_categories(pd.Series(["a"] * 500 + ["b"] * 400 + list("cdefghijkl")))
    assert rare == (("a", "b"), "other")


# ── planning and code generation ─────────────────────────────────────────────

def test_plan_covers_phase2_steps(report):
    text = "\n".join(report.plan().describe())
    for expected in [
        "customer_id: numeric ID column",
        "refund_issued: possible target leakage",
        "drop rows where 'churn' is null",
        "convert 'price' from text to numbers",
        "trim and lower-case 'city'",
        "group rare values of 'source'",
        "add 'income_was_missing' flag",
        "fill nulls in 'income'",
        "fill nulls in 'price'",
    ]:
        assert expected in text, expected
    assert "'churn' with" not in text            # the target itself is never filled


def test_plan_steps_can_be_switched_off(report):
    plan = report.plan(drop_leaks=False, missing_indicators=False, group_rare=False)
    text = "\n".join(plan.describe())
    assert "leakage" not in text and "_was_missing" not in text and "rare" not in text


@pytest.mark.parametrize("eng", RUNNABLE)
def test_generated_code_matches_reference(report, eng):
    result = report.codegen(eng, validate=False).validate()
    assert result.ok and not result.skipped, str(result)


@pytest.mark.parametrize("eng", RUNNABLE)
def test_clean_end_to_end_on_csv(churn, tmp_path, eng):
    src = tmp_path / "churn.csv"
    churn.to_csv(src, index=False)
    out = tmp_path / f"{eng}.parquet"

    code = fe.clean(src, target="churn", engine=eng, output=str(out))
    runpy.run_path(str(code.save(tmp_path / f"clean_{eng}.py")), run_name="__main__")

    cleaned = pd.read_parquet(out)
    assert compare_frames(code.plan.apply(pd.read_csv(src)), cleaned) is None
    assert cleaned["churn"].notna().all()
    assert {"customer_id", "refund_issued"}.isdisjoint(cleaned.columns)
    assert cleaned["income_was_missing"].any()
    assert set(cleaned["city"]) == {"delhi", "pune", "mumbai"}
    assert code.report.target["task"] == "binary"


def test_reports_render_with_target(report, tmp_path):
    assert "Target: churn" in report.to_html()
    assert "possible leak" in report.to_html()
    report.save_pdf(str(tmp_path / "r.pdf"))
    assert report.to_dict()["target"]["task"] == "binary"
