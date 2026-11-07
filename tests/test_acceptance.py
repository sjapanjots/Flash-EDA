"""
Acceptance tests with hand-written expected outputs.

Every other codegen test compares generated code with Plan.apply(), so a
bug in both would pass. Here the expected tables are written out by hand.
"""

import importlib.util
import math

import pandas as pd
import pytest

import flasheda as fe
from flasheda.codegen import generate, run_generated
from flasheda.codegen.steps import (
    DateParts, DropColumns, DropDuplicates, DropNullRows, FillNull, GroupRare, Log1p,
    MissingIndicator, NormalizeText, OneHot, Plan, Ratio, ToNumeric,
)

ENGINES = [e for e in ("pandas", "polars", "duckdb", "pyspark")
           if e == "pandas" or importlib.util.find_spec(e) is not None]


def run(plan, df, engine):
    code = generate(plan, engine, columns=list(df.columns)).code
    return run_generated(code, engine, df.copy()).reset_index(drop=True)


def same(got: pd.DataFrame, expected: dict):
    assert list(got.columns) == list(expected)
    for col, values in expected.items():
        for g, e in zip(got[col].tolist(), values):
            if e is None:
                assert g is None or (isinstance(g, float) and math.isnan(g)) or g is pd.NA, (col, g)
            elif isinstance(e, float):
                assert g == pytest.approx(e, abs=1e-9), (col, g, e)
            else:
                assert g == e, (col, g, e)
        assert len(got) == len(values)


@pytest.mark.parametrize("engine", ENGINES)
def test_row_steps(engine):
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", None], "junk": [0, 0, 0, 0]})
    plan = Plan([DropColumns(("junk",)), DropDuplicates(), DropNullRows("b")])
    out = run(plan, df, engine).sort_values("a").reset_index(drop=True)   # engines may reorder rows
    same(out, {"a": [1, 2], "b": ["x", "y"]})


@pytest.mark.parametrize("engine", ENGINES)
def test_value_steps(engine):
    df = pd.DataFrame({
        "price": [" 10", "2.5", "n/a", "7 "],
        "city": [" Delhi", "DELHI", "Pune", None],
        "size": [1.0, None, 3.0, None],
        "amount": [0.0, 9.0, -9.0, 99.0],
    })
    plan = Plan([
        ToNumeric("price"),
        NormalizeText("city"),
        MissingIndicator("size", "size_was_missing"),
        FillNull("size", 2.0),
        Log1p("amount"),
    ])
    same(run(plan, df, engine), {
        "price": [10.0, 2.5, None, 7.0],
        "city": ["delhi", "delhi", "pune", None],
        "size": [1.0, 2.0, 3.0, 2.0],
        "amount": [0.0, math.log(10), -math.log(10), math.log(100)],
        "size_was_missing": [False, True, False, True],
    })


@pytest.mark.parametrize("engine", ENGINES)
def test_feature_steps(engine):
    df = pd.DataFrame({
        "day": ["2024-03-04", "2024-12-29 18:30:00", None],
        "a": [10.0, 3.0, 5.0],
        "b": [2.0, 0.0, 5.0],
        "color": ["red", "blue", "green"],
    })
    plan = Plan([
        GroupRare("color", ("red", "blue"), "other"),
        DateParts("day", ("year", "dayofweek", "hour"), ("day_year", "day_dayofweek", "day_hour")),
        Ratio("a", "b", "a_per_b"),
        OneHot("color", ("red", "other"), ("color_red", "color_other")),
    ])
    got = run(plan, df, engine)
    same(got[["color", "a_per_b", "color_red", "color_other"]], {
        "color": ["red", "blue", "other"],
        "a_per_b": [5.0, None, 1.0],                 # division by zero → null
        "color_red": [True, False, False],
        "color_other": [False, False, True],
    })
    assert got["day_year"].tolist()[:2] == [2024, 2024]
    assert got["day_dayofweek"].tolist()[:2] == [1, 7]  # Monday = 1, Sunday = 7
    assert got["day_hour"].tolist()[:2] == [0, 18]
    assert pd.isna(got["day_year"].iloc[2])


def test_protected_columns_are_never_dropped_or_changed():
    n = 400
    df = pd.DataFrame({
        "order_id": range(n),                            # an ID: dropped unless protected
        "comment": ["x"] * 300 + [None] * 100,
        "notes": [None] * 300 + ["y"] * 100,             # 75% missing: dropped unless protected
        "price": [str(i) for i in range(n)],             # numbers as text: converted unless protected
    })
    report = fe.analyze(df, save_pdf=False)
    default = report.plan()
    dropped = {c for s in default.steps if isinstance(s, DropColumns) for c in s.columns}
    assert {"order_id", "notes"} <= dropped

    plan = report.plan(protect=["order_id", "notes", "price"])
    out = plan.apply(df)
    for col in ("order_id", "notes", "price"):
        pd.testing.assert_series_equal(out[col], df[col])
    with_features = report.plan(features=True, protect=["order_id", "notes", "price"]).apply(df)
    for col in ("order_id", "notes", "price"):
        pd.testing.assert_series_equal(with_features[col], df[col])
    with pytest.raises(KeyError, match="nope"):
        report.plan(protect=["nope"])
