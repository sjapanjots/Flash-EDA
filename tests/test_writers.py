import ast

import numpy as np
import pandas as pd
import pytest

from flasheda.codegen import (
    ClipOutliers, DropColumns, DropDuplicates, FillNull, Log1p, Plan, compare_frames, generate,
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


@pytest.mark.parametrize("eng", RUNNABLE)
def test_every_step_matches_reference(frame, eng):
    result = generate(FULL_PLAN, eng, sample=frame).validate()
    assert result.ok and not result.skipped, str(result)


def test_pyspark_code_is_valid_python():
    ast.parse(generate(FULL_PLAN, "pyspark", source="data.csv").code)


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
