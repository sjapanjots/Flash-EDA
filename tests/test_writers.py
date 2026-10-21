import ast

import numpy as np
import pandas as pd
import pytest

from flasheda.codegen import (
    ClipOutliers, DateParts, DropColumns, DropDuplicates, DropNullRows, FillNull,
    FrequencyEncode, GroupRare, Log1p, MissingIndicator, NormalizeText, OneHot, Plan, Ratio,
    ToNumeric, compare_frames, generate,
)

RUNNABLE = ["pandas", "polars", "duckdb"]

FULL_PLAN = Plan([
    DropColumns(("id",), reason="likely an ID column"),
    DropDuplicates(reason="duplicates in sample"),
    FillNull("age", 40, reason="median of sample"),
    FillNull("score", 0.5, reason="median of sample"),
    FillNull("city", "O'Hare", reason="mode of sample"),
    ClipOutliers("temp.c", -1.5, 30.25, reason="IQR outliers"),
    Log1p('say "hi"', reason="skewed"),
])


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 2_000
    df = pd.DataFrame({
        "age": np.where(rng.random(n) < 0.1, np.nan, rng.integers(18, 80, n)),
        "score": np.where(rng.random(n) < 0.1, np.nan, rng.random(n)),
        "city": np.where(rng.random(n) < 0.1, None, rng.choice(["Delhi", "Pune"], n)),
        "temp.c": rng.normal(10, 20, n),
        'say "hi"': rng.lognormal(0, 2, n),
    })
    df = pd.concat([df, df.head(50)], ignore_index=True)   # duplicates once "id" is dropped
    return df.assign(id=np.arange(len(df)))


# Phase 2 steps, in the order the planner emits them: a row step in the
# middle (DropDuplicates) and a flag that must see nulls created by ToNumeric.
CLEAN_PLAN = Plan([
    DropNullRows("label", reason="target is null"),
    ToNumeric("price", reason="numbers stored as text"),
    NormalizeText("city", reason="spelling variants"),
    DropDuplicates(reason="duplicates after normalising"),
    GroupRare("city", ("delhi", "pune"), reason="rare values"),
    MissingIndicator("price", "price_was_missing", reason="missingness predicts target"),
    FillNull("price", 9.5, reason="median of sample"),
    Log1p("price", reason="skewed"),
])


@pytest.fixture(scope="module")
def dirty() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 2_000
    price = rng.lognormal(2, 1, n).round(2).astype(str).astype(object)
    price[rng.random(n) < 0.05] = "N/A"
    price[rng.random(n) < 0.05] = None
    price[rng.random(n) < 0.05] = " 12.5 "
    city = rng.choice(["Delhi", " delhi", "DELHI ", "Pune", "pune", "Agra", "Goa"], n).astype(object)
    city[rng.random(n) < 0.05] = None
    label = rng.choice(["yes", "no"], n).astype(object)
    label[rng.random(n) < 0.03] = None
    df = pd.DataFrame({"price": price, "city": city, "label": label})
    return pd.concat([df, df.head(100)], ignore_index=True)


@pytest.mark.parametrize("eng", RUNNABLE)
def test_every_step_matches_reference(frame, eng):
    result = generate(FULL_PLAN, eng, sample=frame).validate()
    assert result.ok and not result.skipped, str(result)


@pytest.mark.parametrize("eng", RUNNABLE)
def test_clean_steps_match_reference(dirty, eng):
    result = generate(CLEAN_PLAN, eng, sample=dirty).validate()
    assert result.ok and not result.skipped, str(result)


def test_clean_steps_reference_semantics(dirty):
    out = CLEAN_PLAN.apply(dirty)
    assert out["label"].notna().all()
    assert set(out["city"].dropna()) == {"delhi", "pune", "other"}
    assert out["price"].notna().all()
    assert out["price_was_missing"].sum() > 0          # includes "N/A" turned into null


# Phase 3 feature steps. FillNull on a date part comes after DateParts in the
# same run of column steps, so it must start a new segment in polars/DuckDB.
FEATURE_PLAN = Plan([
    DateParts("when", ("year", "month", "day", "dayofweek", "hour"),
              ("when_year", "when_month", "when_day", "when_dayofweek", "when_hour")),
    Ratio("a", "b", "a_per_b"),
    OneHot("city", ("delhi", "pune"), ("city_delhi", "city_pune")),
    FrequencyEncode("source", (("web", 0.6), ("app", 0.3)), 0.0),
    FillNull("when_year", 2023),
    DropColumns(("when", "city")),
])


@pytest.fixture(scope="module")
def featured() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 2_000
    when = (pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 10**6, n), unit="min"))
    when = when.strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype(object)
    when[::7] = pd.Series(when[::7]).str[:10].to_numpy()   # some date-only values
    when[rng.random(n) < 0.05] = None
    when[rng.random(n) < 0.02] = "not a date"
    b = rng.integers(0, 5, n).astype(float)
    b[rng.random(n) < 0.05] = np.nan
    return pd.DataFrame({
        "when": when,
        "a": rng.normal(10, 3, n),
        "b": b,
        "city": rng.choice(["delhi", "pune", "goa", None], n),
        "source": rng.choice(["web", "app", "email", None], n),
    })


@pytest.mark.parametrize("eng", RUNNABLE)
def test_feature_steps_match_reference(featured, eng):
    result = generate(FEATURE_PLAN, eng, sample=featured).validate()
    assert result.ok and not result.skipped, str(result)


def test_feature_steps_reference_semantics(featured):
    out = FEATURE_PLAN.apply(featured)
    first = pd.Timestamp(featured["when"].dropna().iloc[0])
    assert out.loc[0, "when_dayofweek"] in range(1, 8)
    assert out["when_year"].notna().all()                       # filled after DateParts
    assert out.loc[featured["b"] == 0, "a_per_b"].isna().all()  # x / 0 → null
    assert out.loc[featured["city"].isna(), ["city_delhi", "city_pune"]].eq(False).all().all()
    assert set(out["source"].unique()) == {0.6, 0.3, 0.0}
    assert {"when", "city"}.isdisjoint(out.columns)
    assert first.isoweekday() in range(1, 8)


@pytest.mark.parametrize("plan", [FULL_PLAN, CLEAN_PLAN, FEATURE_PLAN])
def test_pyspark_code_is_valid_python(plan):
    ast.parse(generate(plan, "pyspark", source="data.csv").code)


@pytest.mark.parametrize("eng", RUNNABLE + ["pyspark"])
def test_empty_plan(eng):
    code = generate(Plan(), eng, source="x.csv").code
    ast.parse(code)
    assert "none — the data looked clean" in code


@pytest.mark.parametrize("eng", RUNNABLE)
def test_log_transform_safe_for_values_unseen_in_sample(eng):
    plan = Plan([Log1p("x")])
    df = pd.DataFrame({"x": [-5.0, -1.0, 0.0, 3.0, None]})
    out = generate(plan, eng).run(df)
    assert out["x"].isna().sum() == 1                     # only the original null
    assert compare_frames(plan.apply(df), out) is None


def test_unknown_engine():
    with pytest.raises(ValueError, match="No code writer"):
        generate(Plan(), "excel")


def test_compare_frames_detects_differences():
    a = pd.DataFrame({"x": [1.0, 2.0]})
    assert compare_frames(a, a[::-1]) is None
    assert "differ" in compare_frames(a, pd.DataFrame({"x": [1.0, 3.0]}))
    assert "row count" in compare_frames(a, a.head(1))
