import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.codegen import Plan
from flasheda.codegen.steps import DropColumns, DropDuplicates, FillNull, MissingIndicator

N = 20_000


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "id": np.arange(N),
        "age": rng.integers(18, 80, N).astype(float),
        "city": rng.choice(["a", "b", "c"], N),
        "empty": [None] * N,
    })
    df.loc[rng.choice(N, 3000, replace=False), "age"] = np.nan
    return pd.concat([df, df.iloc[:500]], ignore_index=True)       # 500 duplicate rows


PLAN = Plan([
    DropColumns(("empty",)),
    DropDuplicates(),
    MissingIndicator("age", "age_was_missing"),
    FillNull("age", 40.0),
])


def test_dataframe_dry_run_is_exact(data):
    imp = PLAN.impact(data)
    assert (imp.engine, imp.exact) == ("pandas", True)
    assert imp.rows_before == N + 500 and imp.rows_after == N and imp.rows_removed == 500
    assert imp.dropped_columns == ["empty"] and imp.added_columns == ["age_was_missing"]
    age = imp.columns.set_index("column").loc["age"]
    assert age["nulls_before"] > 3000 and age["nulls_after"] == 0
    assert imp.to_dict()["rows_after"] == N


@pytest.mark.parametrize("ext", ["parquet", "csv"])
def test_file_dry_run_matches_pandas(data, tmp_path, ext):
    path = tmp_path / f"d.{ext}"
    data.drop(columns="empty").assign(empty=None).pipe(
        lambda d: d.to_parquet(path, index=False) if ext == "parquet" else d.to_csv(path, index=False))
    imp = PLAN.impact(path)
    exact = PLAN.impact(pd.read_parquet(path) if ext == "parquet" else pd.read_csv(path))
    assert imp.engine == "duckdb" and not imp.exact
    assert (imp.rows_before, imp.rows_after) == (exact.rows_before, exact.rows_after)
    assert imp.dropped_columns == ["empty"] and imp.added_columns == ["age_was_missing"]
    a, e = imp.columns.set_index("column"), exact.columns.set_index("column")
    assert a.loc["age", "nulls_before"] == e.loc["age", "nulls_before"]
    assert a.loc["age", "nulls_after"] == 0


def test_report_plan_dry_run_on_the_source(data, tmp_path, capsys):
    path = tmp_path / "d.parquet"
    data.drop(columns="empty").to_parquet(path, index=False)
    report = fe.analyze(path, save_pdf=False)
    imp = report.plan().impact(path)
    assert imp.rows_before == N + 500
    imp.show()
    assert "Full-data dry run" in capsys.readouterr().out


def test_bad_source():
    with pytest.raises(TypeError):
        PLAN.impact(42)
    with pytest.raises(FileNotFoundError):
        PLAN.impact("missing.parquet")
