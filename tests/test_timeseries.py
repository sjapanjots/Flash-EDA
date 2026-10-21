import numpy as np
import pandas as pd
import polars as pl
import pytest

import flasheda as fe
from flasheda.timeseries import aggregate


def make_orders(growth=0.3, weekend=1.6, seed=0, start="2023-01-01", end="2025-12-31"):
    """Daily orders: yearly growth, busier weekends, a 7-day outage and one spike day."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    rate = 200 * (1 + growth * np.arange(len(days)) / 365) * np.where(days.dayofweek >= 5, weekend, 1.0)
    counts = rng.poisson(rate)
    if len(counts) > 700:
        counts[400:407] = 0
        counts[700] *= 8
    ts = np.repeat(days.values, counts) + pd.to_timedelta(rng.integers(0, 86_400, counts.sum()), unit="s").values
    return pd.DataFrame({"created_at": ts, "amount": rng.lognormal(3, 0.5, counts.sum())})


@pytest.fixture(scope="module")
def orders():
    return make_orders()


@pytest.fixture(scope="module")
def ts(orders):
    return fe.timeseries(orders, time="created_at")


def test_profile(ts, orders):
    assert ts.freq == "day" and ts.n_periods == 1096
    assert ts.start == pd.Timestamp("2023-01-01") and ts.end == pd.Timestamp("2025-12-31")
    assert ts.series["rows"].sum() == len(orders)
    assert ts.missing_periods == 7 and ts.longest_gap == 7
    assert ts.trend["direction"] == "rising" and 10 < ts.trend["change_per_year_pct"] < 30
    assert ts.season_length == 7
    assert max(ts.profile, key=ts.profile.get) in ("Saturday", "Sunday")
    assert ts.anomalies[0]["period"].startswith(str(pd.Timestamp("2023-01-01") + pd.Timedelta(days=700))[:10])


@pytest.mark.parametrize("kind", ["parquet", "csv", "polars", "sql"])
def test_sources_agree(orders, ts, tmp_path, kind):
    if kind == "parquet":
        orders.to_parquet(tmp_path / "o.parquet")
        src = tmp_path / "o.parquet"
    elif kind == "csv":
        orders.assign(created_at=orders["created_at"].astype(str)).to_csv(tmp_path / "o.csv", index=False)
        src = tmp_path / "o.csv"
    elif kind == "polars":
        src = pl.from_pandas(orders)
    else:
        sa = pytest.importorskip("sqlalchemy")
        url = f"sqlite:///{tmp_path / 'o.db'}"
        orders.assign(created_at=orders["created_at"].astype(str)).to_sql("o", sa.create_engine(url), index=False)
        src = fe.sql(url, table="o")
    other = fe.timeseries(src, time="created_at")
    pd.testing.assert_series_equal(other.series["rows"], ts.series["rows"])
    assert other.trend["direction"] == ts.trend["direction"] and other.season_length == 7


def test_value_sum_and_mean(orders):
    s = fe.timeseries(orders, time="created_at", value="amount", freq="week")
    assert s.freq == "week" and s.metric == "sum(amount)"
    assert s.series["value"].sum() == pytest.approx(orders["amount"].sum())
    m = fe.timeseries(orders, time="created_at", value="amount", agg="mean")
    assert m.trend["direction"] == "flat"                   # average order size doesn't change
    assert m.series["value"].isna().sum() == 7              # empty days have no mean


def test_falling_and_flat():
    assert fe.timeseries(make_orders(growth=-0.25), time="created_at").trend["direction"] == "falling"
    flat = fe.timeseries(make_orders(growth=0, weekend=1.0), time="created_at")
    assert flat.trend["direction"] == "flat" and flat.season_length is None


def test_auto_frequency():
    hourly = make_orders(start="2024-01-01", end="2024-01-20")
    assert fe.timeseries(hourly, time="created_at").freq == "hour"
    weekly = make_orders(start="2015-01-01", end="2024-12-31")
    assert fe.timeseries(weekly, time="created_at").freq == "week"


def test_week_buckets_start_on_monday(orders):
    series, _ = aggregate(orders, "created_at", freq="week")
    assert (series["period"].dt.dayofweek == 0).all()


def test_bad_arguments(orders):
    with pytest.raises(ValueError, match="agg"):
        fe.timeseries(orders, time="created_at", agg="max")
    with pytest.raises(ValueError, match="freq"):
        fe.timeseries(orders, time="created_at", freq="month")


def test_show_and_dict(ts, capsys):
    ts.show()
    out = capsys.readouterr().out
    assert "rising" in out and "weekly seasonality" in out
    assert ts.to_dict()["trend"]["direction"] == "rising"
