import importlib.util

import pandas as pd
import pytest

import flasheda as fe
from flasheda.clean.detectors import date_format
from flasheda.codegen import generate, run_generated
from flasheda.codegen.steps import DateParts, Plan, java_date_format

ENGINES = [e for e in ("pandas", "polars", "duckdb")
           if e == "pandas" or importlib.util.find_spec(e) is not None]


def col(values, n=60):
    return pd.Series((values * n)[:n], dtype=object)


@pytest.mark.parametrize("values, fmt", [
    (["25/12/2026", "07/11/2026", "13/01/2025"], "%d/%m/%Y"),
    (["12/25/2026", "07/11/2026", "01/13/2025"], "%m/%d/%Y"),
    (["7-Nov-2026", "25-Dec-2026"], "%d-%b-%Y"),
    (["25.12.2026 18:30", "07.11.2026 09:05"], "%d.%m.%Y %H:%M"),
])
def test_unambiguous_formats_are_detected(values, fmt):
    found = date_format(col(values))
    assert found is not None and found.format == fmt and not found.ambiguous


@pytest.mark.parametrize("values", [
    ["07/11/2026", "03/04/2026"],          # day/month or month/day?
    ["07-11-26", "03-04-25"],              # and the year could be first too
])
def test_ambiguous_dates_are_not_guessed(values):
    found = date_format(col(values))
    assert found.ambiguous and found.alternatives


def test_non_dates():
    assert date_format(col(["Delhi", "Pune"])) is None
    assert date_format(col(["1,250", "3"])) is None


def test_java_pattern():
    assert java_date_format("%d/%m/%Y %H:%M") == "d/M/yyyy H:m"
    assert java_date_format("%dT%m") == "d'T'M"


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_parses_formatted_dates(engine):
    df = pd.DataFrame({"day": ["25/12/2026", "07/11/2026", "bad", None]})
    plan = Plan([DateParts("day", ("year", "month", "day", "dayofweek"),
                           ("y", "m", "d", "dow"), True, "%d/%m/%Y")])
    out = run_generated(generate(plan, engine, columns=["day"]).code, engine, df.copy())
    assert out["m"].tolist()[:2] == [12, 11] and out["d"].tolist()[:2] == [25, 7]
    assert out["dow"].tolist()[:2] == [5, 6]           # Friday, Saturday (ISO)
    assert out["y"].iloc[2:].isna().all()


def test_planner_uses_formats_and_flags_ambiguity():
    days = pd.date_range("2025-01-01", periods=400, freq="D")
    early = days[days.day <= 12][:72]                                   # no day above 12
    df = pd.DataFrame({
        "shipped": days[:72].strftime("%d/%m/%Y"),                      # has days above 12
        "ordered": early.strftime("%m/%d/%Y"),
        "y": [i % 2 for i in range(72)],
    })
    report = fe.analyze(df, target="y", save_pdf=False)
    plan = report.plan(features=True)
    parts = {s.column: s for s in plan.steps if isinstance(s, DateParts)}
    assert parts["shipped"].format == "%d/%m/%Y" and "ordered" not in parts
    assert any(s.startswith("ordered:") and "date_formats" in s for s in plan.suggestions)

    chosen = report.plan(features=True, date_formats={"ordered": "%m/%d/%Y"})
    parts = {s.column: s for s in chosen.steps if isinstance(s, DateParts)}
    assert parts["ordered"].format == "%m/%d/%Y"
    month = chosen.apply(df)[parts["ordered"].names[parts["ordered"].parts.index("month")]]
    assert month.iloc[:3].tolist() == [1, 1, 1] and month.max() == 6
    with pytest.raises(ValueError, match="features=True"):
        report.plan(date_formats={"ordered": "%m/%d/%Y"})
    with pytest.raises(KeyError):
        report.plan(features=True, date_formats={"nope": "%d"})
