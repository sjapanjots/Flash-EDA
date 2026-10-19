import ast

import numpy as np
import pandas as pd
import pytest

from flasheda.codegen import (
    ClipOutliers, DropColumns, DropDuplicates, DropNullRows, FillNull, GroupRare, Log1p,
    MissingIndicator, NormalizeText, Plan, ToNumeric, compare_frames, generate,
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


@pytest.mark.parametrize("plan", [FULL_PLAN, CLEAN_PLAN])
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
