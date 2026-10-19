import numpy as np
import pandas as pd
import pytest


def make_messy(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """A dataset with every problem the planner knows how to fix."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "user id": [f"u{i}" for i in range(n)],
        "age": np.where(rng.random(n) < 0.1, np.nan, rng.integers(18, 80, n)),
        "income": rng.lognormal(10, 1, n),
        "temp.c": np.r_[rng.normal(20, 2, n - n // 10), rng.normal(-60, 5, n // 10)],
        "city": np.where(rng.random(n) < 0.05, None, rng.choice(["Delhi", "Pune", "O'Hare"], n)),
        'say "hi"': rng.choice(["a", "b"], n),
        "mostly_null": np.where(rng.random(n) < 0.8, np.nan, rng.random(n)),
        "flag": rng.random(n) < 0.5,
    })
    return pd.concat([df, df.iloc[: n // 50]], ignore_index=True)


@pytest.fixture(scope="session")
def messy() -> pd.DataFrame:
    return make_messy()


@pytest.fixture(scope="session")
def files(tmp_path_factory, messy):
    d = tmp_path_factory.mktemp("data")
    paths = {"csv": d / "messy.csv", "tsv": d / "messy.tsv", "parquet": d / "messy.parquet"}
    messy.to_csv(paths["csv"], index=False)
    messy.to_csv(paths["tsv"], index=False, sep="\t")
    messy.to_parquet(paths["parquet"], index=False)
    return paths
