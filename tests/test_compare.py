import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.compare import psi
from test_clean_target import make_churn


@pytest.fixture(scope="module")
def jan():
    return make_churn(12_000, seed=1)


@pytest.fixture(scope="module")
def feb():
    return make_churn(12_000, seed=2)


def test_psi_basics():
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0
    assert psi([0.5, 0.5], [0.4, 0.6]) == pytest.approx(0.0405, abs=1e-3)
    assert psi([0.9, 0.1], [0.1, 0.9]) > 1


def test_same_distribution_is_stable(jan, feb, tmp_path):
    jan.to_csv(tmp_path / "jan.csv", index=False)
    feb.to_parquet(tmp_path / "feb.parquet", index=False)
    for ref, cur in [(jan, feb), (tmp_path / "jan.csv", tmp_path / "feb.parquet")]:
        drift = fe.compare(ref, cur)
        assert drift.ok
        assert drift.drifted == [], [(c["column"], c["psi"]) for c in drift.drifted]
        assert drift.rows_reference == drift.rows_current == 12_100


def test_numeric_shift(jan, feb):
    drift = fe.compare(jan, feb.assign(income=feb["income"] * 2))
    c = drift["income"]
    assert c["severity"] == "major" and c["median_shift_std"] > 0.5 and c["ks_pvalue"] < 1e-6
    assert drift.drifted[0]["column"] == "income"          # sorted by PSI


def test_new_categories(jan, feb):
    cur = feb.copy()
    cur.loc[: len(cur) // 5, "city"] = "Paris"
    c = fe.compare(jan, cur)["city"]
    assert c["severity"] == "major"
    assert c["new_values"] == ["Paris"] and c["new_values_pct"] == pytest.approx(20, abs=2)


def test_null_jump_is_drift(jan, feb):
    cur = feb.copy()
    cur.loc[: len(cur) // 4, "income"] = np.nan
    assert fe.compare(jan, cur)["income"]["severity"] != "stable"


def test_schema_changes(jan, feb):
    cur = feb.drop(columns="source").assign(extra=1, churn=feb["churn"].notna())
    drift = fe.compare(jan, cur)
    assert not drift.ok
    assert drift.added_columns == ["extra"] and drift.removed_columns == ["source"]
    assert drift.type_changes == [{"column": "churn", "reference": "text", "current": "bool"}]


def test_dates_and_ids_are_not_noise(jan, feb):
    later = feb.assign(signup_date=pd.to_datetime(feb["signup_date"]) + pd.Timedelta(days=365))
    drift = fe.compare(jan, later.assign(signup_date=later["signup_date"].astype(str)))
    assert drift["signup_date"]["severity"] == "stable"
    assert "→" in drift["signup_date"]["detail"]
    assert "new_values" not in drift["customer_id"]


def test_show_and_dict(jan, feb, capsys):
    drift = fe.compare(jan, feb)
    drift.show()
    assert "no major drift" in capsys.readouterr().out
    assert drift.to_dict()["ok"] is True
