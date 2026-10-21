import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.timeseries.forecast import METHODS
from test_timeseries import make_orders


@pytest.fixture(scope="module")
def orders(tmp_path_factory):
    path = tmp_path_factory.mktemp("fc") / "orders.parquet"
    make_orders().to_parquet(path)
    return path


@pytest.fixture(scope="module")
def result(orders):
    return fe.timeseries(orders, time="created_at").forecast()


def test_methods():
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    assert METHODS["naive"](y, 3, None).tolist() == [8, 8, 8]
    assert METHODS["seasonal_naive"](y, 5, 4).tolist() == [5, 6, 7, 8, 5]
    assert METHODS["moving_average"](y, 2, 4).tolist() == [6.5, 6.5]
    assert METHODS["drift"](y, 2, None).tolist() == [9, 10]
    season = np.tile([0.0, 10.0], 100) + np.arange(200)       # trend + alternating season
    assert METHODS["trend_seasonal"](season, 2, 2) == pytest.approx([200, 211], abs=0.1)


def test_seasonal_data_picks_seasonal_method(result):
    assert result.method in ("trend_seasonal", "seasonal_naive")
    board = result.leaderboard.set_index("method")["mae"]
    assert board[result.method] < 0.5 * board["naive"]
    assert result.horizon == 14 and result.season == 7
    fc = result.forecast
    assert fc["period"].iloc[0] == pd.Timestamp("2026-01-01") and len(fc) == 14
    weekend = fc["period"].dt.dayofweek >= 5
    assert fc.loc[weekend, "forecast"].mean() > 1.3 * fc.loc[~weekend, "forecast"].mean()


def test_random_walk_prefers_simple_methods():
    rng = np.random.default_rng(3)
    days = pd.date_range("2024-01-01", periods=400, freq="D")
    df = pd.DataFrame({"day": days, "price": 100 + rng.normal(0, 1, 400).cumsum()})
    res = fe.timeseries(df, time="day", value="price", agg="mean").forecast(horizon=7)
    assert res.season is None
    assert set(res.leaderboard["method"]) == {"naive", "moving_average", "drift"}   # no season


def test_errors(orders):
    ts = fe.timeseries(orders, time="created_at")
    with pytest.raises(ValueError, match="too short"):
        ts.forecast(horizon=400)
    short = fe.timeseries(make_orders(growth=0, weekend=1.0), time="created_at")
    with pytest.raises(ValueError, match="No usable methods"):
        short.forecast(methods=["seasonal_naive"])


def _run(script, data, block_duckdb=False):
    code = ("import sys, runpy\n"
            + ("sys.modules['duckdb'] = None\n" if block_duckdb else "")
            + f"sys.argv = ['forecast.py', {str(data)!r}]\n"
            + f"runpy.run_path({str(script)!r}, run_name='__main__')\n")
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
@pytest.mark.parametrize("block_duckdb", [False, True])
def test_script_reproduces_forecast(result, orders, tmp_path, fmt, block_duckdb):
    data = orders
    if fmt == "csv":
        df = pd.read_parquet(orders)
        data = tmp_path / "orders.csv"
        df.assign(created_at=df["created_at"].astype(str)).to_csv(data, index=False)
    out = tmp_path / "fc.csv"
    script = tmp_path / "forecast.py"
    script.write_text(result.script(output=str(out)))

    res = _run(script, data, block_duckdb)
    assert res.returncode == 0, res.stderr
    got = pd.read_csv(out, parse_dates=["period"])
    pd.testing.assert_series_equal(got["period"], result.forecast["period"])
    np.testing.assert_allclose(got["forecast"], result.forecast["forecast"], rtol=1e-9)


def test_show(result, capsys):
    result.show()
    assert result.method in capsys.readouterr().out
