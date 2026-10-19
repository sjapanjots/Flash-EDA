import ast
import runpy

import pandas as pd
import pytest

import flasheda as fe
from flasheda.codegen import compare_frames

RUNNABLE = ["pandas", "polars", "duckdb"]


@pytest.fixture(scope="module")
def report(messy):
    return fe.analyze(messy, save_pdf=False)


def test_plan_covers_every_problem(report):
    text = "\n".join(report.plan().describe())
    assert "user id" in text and "mostly_null" in text          # dropped columns
    assert "drop duplicate rows" in text
    assert "fill nulls in 'age'" in text                        # numeric median
    assert "fill nulls in 'city'" in text                       # text mode
    assert "log transform 'income'" in text                   # skew
    assert "clip 'temp.c'" in text                              # outliers


def test_plan_options_switch_steps_off(report):
    plan = report.plan(fill_nulls=False, clip_outliers=False, drop_duplicates=False)
    text = "\n".join(plan.describe())
    assert "fill" not in text and "clip" not in text and "duplicate" not in text


@pytest.mark.parametrize("eng", RUNNABLE)
def test_generated_code_matches_reference(report, eng):
    result = report.codegen(eng, validate=False).validate()
    assert result.ok and not result.skipped, str(result)


def test_pyspark_code_is_valid_python(report):
    ast.parse(report.codegen("pyspark", validate=False).code)


@pytest.mark.parametrize("eng", RUNNABLE)
@pytest.mark.parametrize("kind", ["csv", "parquet"])
def test_generated_script_runs_end_to_end(files, tmp_path, eng, kind):
    report = fe.analyze(files[kind], save_pdf=False)
    out = tmp_path / f"{eng}.parquet"
    script = report.codegen(eng, output=str(out)).save(tmp_path / f"clean_{eng}.py")

    runpy.run_path(str(script), run_name="__main__")

    expected = report.plan().apply(pd.read_parquet(files["parquet"]))
    assert compare_frames(expected, pd.read_parquet(out)) is None


def test_analyze_file_reports_full_row_count(files, messy):
    report = fe.analyze(files["csv"], n=500, save_pdf=False)
    assert report.original_shape == messy.shape
    assert report.overview["original_rows"] == len(messy)
    assert report.sample_size == 500
    assert report.source == str(files["csv"])


def test_engine_override(messy):
    assert fe.analyze(messy, save_pdf=False, engine="duckdb").recommended_engine == "duckdb"
