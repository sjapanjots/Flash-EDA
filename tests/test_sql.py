import os
import runpy

import pandas as pd
import pytest

import flasheda as fe
from flasheda.codegen import compare_frames
from test_clean_target import make_churn

sa = pytest.importorskip("sqlalchemy")


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "shop.db"
    url = f"sqlite:///{path}"
    df = make_churn(20_000, seed=4).astype({"refund_issued": int})   # SQLite has no booleans
    df.to_sql("customers", sa.create_engine(url), index=False)
    return url, df


def test_table_source(db):
    url, df = db
    report = fe.analyze(fe.sql(url, table="customers"), target="churn", save_pdf=False)
    assert report.original_shape == df.shape            # count(*) in the database
    assert report.sample_size == 5_000                  # only the sample is transferred
    assert report.recommended_engine == "pandas"
    assert report.target["leakage_columns"][0]["column"] == "refund_issued"
    assert "(table customers)" in report.to_dict()["source"]


def test_query_source(db):
    url, df = db
    src = fe.sql(url, query="SELECT customer_id, income, churn FROM customers WHERE income > 20000;")
    report = fe.analyze(src, save_pdf=False)
    assert report.original_shape == ((df["income"] > 20_000).sum(), 3)


def test_password_never_leaks():
    src = fe.sql("postgresql://bob:s3cret@db.example.com/shop", table="public.orders")
    assert "s3cret" not in src.display and "***" in src.display
    assert src.from_sql() == "public.orders"
    with pytest.raises(ValueError, match="exactly one"):
        fe.sql("sqlite://", table="a", query="SELECT 1")


def test_generated_script_reads_url_from_env(db, tmp_path, monkeypatch):
    url, df = db
    report = fe.analyze(fe.sql(url, table="customers"), target="churn", save_pdf=False)
    out = tmp_path / "clean.parquet"
    code = report.codegen(output=str(out))
    assert code.validate().ok
    assert "url = os.environ['DATABASE_URL']" in code.code
    assert "create_engine(url)" in code.code and "create_engine('" not in code.code

    monkeypatch.setenv("DATABASE_URL", url)
    runpy.run_path(str(code.save(tmp_path / "prep.py")), run_name="__main__")
    with sa.create_engine(url).connect() as con:
        full = pd.read_sql("SELECT * FROM customers", con)
    assert compare_frames(code.plan.apply(full), pd.read_parquet(out)) is None


def test_other_engines_explain_how_to_export(db):
    url, _ = db
    report = fe.analyze(fe.sql(url, table="customers"), save_pdf=False)
    with pytest.raises(ValueError, match="to_parquet"):
        report.codegen("polars")


def test_export_contract_and_compare(db, tmp_path):
    url, df = db
    src = fe.sql(url, table="customers")
    path = src.to_parquet(tmp_path / "customers.parquet", chunksize=3_000)
    exported = pd.read_parquet(path)
    assert exported.shape == df.shape

    contract = fe.analyze(src, save_pdf=False).contract()
    assert fe.check(src, contract).ok
    assert fe.check(path, contract).ok
    assert fe.compare(src, path).ok
