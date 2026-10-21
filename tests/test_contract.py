import importlib

import numpy as np
import pandas as pd
import polars as pl
import pytest

import flasheda as fe
from flasheda.contract import Contract

check_mod = importlib.import_module("flasheda.contract.check")  # the module, not fe.check
from test_clean_target import make_churn


@pytest.fixture(scope="module")
def months(tmp_path_factory):
    d = tmp_path_factory.mktemp("months")
    jan, feb = make_churn(12_000, seed=1), make_churn(12_000, seed=2)
    jan.to_csv(d / "jan.csv", index=False)
    feb.to_csv(d / "feb.csv", index=False)
    feb.to_parquet(d / "feb.parquet", index=False)
    return d


@pytest.fixture(scope="module")
def contract(months):
    return fe.analyze(months / "jan.csv", save_pdf=False).contract()


def feb_frame(months):
    return pd.read_csv(months / "feb.csv")


# ── learning ─────────────────────────────────────────────────────────────────

def test_learned_rules(contract):
    assert contract["customer_id"].unique and contract["customer_id"].max is None   # IDs: no range
    assert contract["income"].kind == "number" and contract["income"].min == 0
    assert contract["price"].kind == "number"           # numbers stored as text, with "N/A"
    assert contract["refund_issued"].kind == "bool"     # "True"/"False" text in the CSV
    assert contract["signup_date"].kind == "datetime"
    assert set(contract["city"].allowed) == {"Delhi", "delhi ", "DELHI", "Pune", "pune", "Mumbai"}
    assert contract["churn"].allowed == ("no", "yes")
    assert contract["source"].allowed is None           # too many values for an allowed set


def test_save_and_load(contract, tmp_path):
    path = contract.save(tmp_path / "c.json")
    assert Contract.load(path) == contract


# ── checking new data that looks like the old data ───────────────────────────

@pytest.mark.parametrize("kind", ["csv", "parquet", "pandas", "polars"])
def test_similar_data_passes(months, contract, kind):
    source = {
        "csv": months / "feb.csv",
        "parquet": months / "feb.parquet",
        "pandas": feb_frame(months),
        "polars": pl.from_pandas(feb_frame(months).astype({"refund_issued": str})),
    }[kind]
    result = fe.check(source, contract)
    assert result.ok, [str(f) for f in result.failures]
    assert result.rows == 12_100


# ── checking broken data ─────────────────────────────────────────────────────

def _break(df: pd.DataFrame, how: str) -> pd.DataFrame:
    df = df.copy()
    if how == "missing column":
        return df.drop(columns="source")
    if how == "duplicate id":
        df.loc[0, "customer_id"] = df.loc[1, "customer_id"]
    elif how == "mostly empty":
        df.loc[: len(df) // 3, "income"] = np.nan
    elif how == "new category":
        df.loc[: len(df) // 20, "city"] = "Paris"
    elif how == "junk values":
        df["price"] = df["price"].astype(object)
        df.loc[: len(df) // 10, "price"] = "unknown"
    elif how == "unit change":
        df["income"] *= 100
    return df


EXPECTED_FAILURE = {
    "missing column": {(None, "required columns")},
    "duplicate id": {("customer_id", "duplicate values")},
    "mostly empty": {("income", "null or unparseable %")},
    "new category": {("city", "unexpected values %")},
    # junk counts as unparseable, and also toward "null or unparseable"
    "junk values": {("price", "values parse as number (% invalid)"),
                    ("price", "null or unparseable %")},
    "unit change": {("income", "maximum")},
}


@pytest.mark.parametrize("how", list(EXPECTED_FAILURE))
@pytest.mark.parametrize("via", ["duckdb", "pandas"])
def test_breaks_are_caught(months, contract, tmp_path, how, via):
    broken = _break(pd.read_csv(months / "feb.csv", dtype={"price": str}), how)
    if via == "duckdb":
        path = tmp_path / "broken.csv"
        broken.to_csv(path, index=False)
        result = fe.check(path, contract)
    else:
        result = fe.check(broken, contract)
    failed = {(f.column, f.rule) for f in result.failures}
    assert failed == EXPECTED_FAILURE[how]


def test_extra_columns(months, contract):
    df = feb_frame(months).assign(extra=1)
    assert fe.check(df, contract).ok
    strict = Contract(contract.columns, allow_extra_columns=False)
    assert [f.rule for f in fe.check(df, strict).failures] == ["no extra columns"]


def test_file_check_without_duckdb(months, contract, monkeypatch):
    monkeypatch.setattr(check_mod, "is_available", lambda e: False)
    assert fe.check(months / "feb.csv", contract).ok


def test_check_accepts_saved_contract(months, contract, tmp_path):
    path = contract.save(tmp_path / "c.json")
    assert fe.check(months / "feb.csv", str(path)).ok
