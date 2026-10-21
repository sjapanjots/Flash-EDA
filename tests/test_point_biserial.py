import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import flasheda as fe
from flasheda.analyzers import correlation


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 2_000
    spend = rng.normal(100, 20, n)
    return pd.DataFrame({
        "spend": spend,
        "noise": rng.normal(0, 1, n),
        "big_spender": spend > 100,                                          # bool dtype
        "big_spender_txt": np.where(spend > 100, "True", "False"),           # text from a CSV
        "coin": rng.random(n) < 0.5,
    })


def test_point_biserial_pairs(df):
    pairs = {(p["numeric"], p["boolean"]): p["point_biserial_r"]
             for p in correlation.analyze(df)["numeric_boolean_strong_pairs"]}
    assert set(pairs) == {("spend", "big_spender"), ("spend", "big_spender_txt")}
    expected = pearsonr(df["spend"], df["big_spender"].astype(float)).statistic
    assert pairs[("spend", "big_spender")] == pytest.approx(expected, abs=1e-4)


def test_nulls_and_constants_are_safe():
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0] * 20,
                       "flag": [True, False, True, None, False] * 20,
                       "const": [True] * 100})
    out = correlation.analyze(df)["numeric_boolean_strong_pairs"]
    assert all(p["boolean"] != "const" for p in out)


def test_warning(df):
    assert any("point-biserial" in w for w in fe.analyze(df).warnings)
