import importlib.util

import pandas as pd
import pytest

import flasheda as fe
from flasheda.clean.detectors import number_format
from flasheda.codegen import generate, run_generated
from flasheda.codegen.steps import Plan, ToNumeric

ENGINES = [e for e in ("pandas", "polars", "duckdb", "pyspark")
           if e == "pandas" or importlib.util.find_spec(e) is not None]


def col(values, n=60):
    return pd.Series((values * n)[:n], dtype=object)


@pytest.mark.parametrize("values, thousands, decimal, strip", [
    (["1,250.50", "980.25", "12,000"], ",", ".", ""),
    (["₹5,000", "₹1,25,000", "₹750"], ",", ".", "₹"),
    (["$12.50", "$3.99", "-$4.00"], ",", ".", "$"),
    (["1.250,50", "980,25", "12.000,1"], ".", ",", ""),
    (["€1.250,50", "€3,5"], ".", ",", "€"),
    (["12%", "7.5%", "100%"], ",", ".", "%"),
])
def test_formats_are_detected(values, thousands, decimal, strip):
    fmt = number_format(col(values))
    assert fmt is not None and not fmt.ambiguous
    assert (fmt.thousands, fmt.decimal, fmt.strip) == (thousands, decimal, strip)


def test_ambiguous_and_non_numbers():
    fmt = number_format(col(["1,250", "2,500", "3,750"]))
    assert fmt.ambiguous                                  # 1250 or 1.25?
    assert number_format(col(["1,250", "2,500"]), decimal=",").decimal == ","
    assert number_format(col(["1,250", "2,500"]), decimal=".").thousands == ","
    assert number_format(col(["Delhi", "Pune"])) is None
    assert number_format(col(["00123", "00456"])) is None   # codes keep leading zeros
    assert number_format(col(["07/11/2026"])) is None


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_parses_the_same(engine):
    df = pd.DataFrame({
        "us": ["1,250.50", "₹1,25,000", "-$4", "n/a"],
        "eu": ["1.250,50", "3,5", "€12", None],
    })
    plan = Plan([ToNumeric("us", ",", ".", "₹$"), ToNumeric("eu", ".", ",", "€")])
    code = generate(plan, engine, columns=list(df.columns)).code
    out = run_generated(code, engine, df.copy())
    assert out["us"].tolist()[:3] == [1250.5, 125000.0, -4.0] and pd.isna(out["us"].iloc[3])
    assert out["eu"].tolist()[:3] == [1250.5, 3.5, 12.0] and pd.isna(out["eu"].iloc[3])
    pd.testing.assert_frame_equal(out.reset_index(drop=True), plan.apply(df), check_dtype=False)


def test_planner_converts_and_flags_ambiguity():
    df = pd.DataFrame({
        "price": col(["₹5,000", "₹1,250.50", "₹980"]),
        "qty": col(["1,250", "2,500", "3,750"]),
    })
    report = fe.analyze(df, save_pdf=False)
    plan = report.plan()
    steps = {s.column: s for s in plan.steps if isinstance(s, ToNumeric)}
    assert steps["price"].strip == "₹" and "qty" not in steps
    assert any(s.startswith("qty:") and "decimal=" in s for s in plan.suggestions)
    assert plan.apply(df)["price"].iloc[:3].tolist() == [5000.0, 1250.5, 980.0]
    chosen = report.plan(decimal=".")
    assert chosen.apply(df)["qty"].iloc[:3].tolist() == [1250.0, 2500.0, 3750.0]
    with pytest.raises(ValueError, match="decimal"):
        report.plan(decimal=";")
