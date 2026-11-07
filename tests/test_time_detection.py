import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.analyzers.target import find_time_columns

N = 300
ISO = pd.date_range("2024-01-01", periods=N, freq="h").strftime("%Y-%m-%d %H:%M:%S")


@pytest.mark.parametrize("name", ["moment", "when", "ts", "period"])
def test_time_column_found_without_a_name_hint(name):
    df = pd.DataFrame({name: ISO})
    assert find_time_columns(df) == [name]


def test_datetime_dtype_and_name_hints_still_work():
    df = pd.DataFrame({"x": pd.to_datetime(ISO), "created_at": list(ISO)})
    assert find_time_columns(df) == ["x", "created_at"]


@pytest.mark.parametrize("values", [
    ["1-2", "3-4", "5-6"] * 100,                  # codes that pandas could read as dates
    ["2024-01-01"] * N,                           # a single constant value
    [f"SKU-{i}" for i in range(N)],
    list(np.arange(N).astype(str)),
])
def test_codes_without_a_name_hint_are_not_dates(values):
    assert find_time_columns(pd.DataFrame({"code": values})) == []


def test_analyze_recommends_time_split_for_unhinted_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"moment": ISO, "x": rng.normal(size=N), "y": rng.integers(0, 2, N)})
    report = fe.analyze(df, target="y", save_pdf=False)
    assert report.target["time_columns"] == ["moment"]
    assert "time-based split on moment" in report.target["recommended_split"]
