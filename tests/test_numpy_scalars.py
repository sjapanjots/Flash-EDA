"""Generated scripts must not contain numpy reprs such as np.str_('b') or np.float64(1.5):
the scripts don't import numpy under that name, and the values must be plain literals."""
import re

import numpy as np
import pandas as pd

import flasheda as fe

NUMPY_REPR = re.compile(r"\bnp\.(str_|float\d*|int\d*|bool_)\(")


def numpy_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(["no"] * (n - 400) + ["yes"] * 400)
    rng.shuffle(labels)
    city = np.array(["Pune", "Delhi", "Goa"])[rng.integers(0, 3, n)].astype(object)
    city[::17] = None
    return pd.DataFrame({
        "amount": rng.lognormal(3, 1, n),
        "city": list(city),                       # numpy str_ objects in an object column
        "y": list(labels),
    })


def test_target_classes_are_plain_strings():
    report = fe.analyze(numpy_frame(), target="y")
    assert all(type(c["value"]) is str for c in report.target["classes"])


def test_generated_scripts_have_no_numpy_reprs():
    df = numpy_frame()
    report = fe.analyze(df, target="y")
    for engine in ("pandas", "polars", "duckdb", "pyspark"):
        assert not NUMPY_REPR.search(report.codegen(engine, features=True, validate=False).code), engine
    result = fe.train(df, target="y", models=["baseline", "logistic_regression"])
    assert not NUMPY_REPR.search(result.train_code)
    assert "POSITIVE = 'yes'" in result.train_code
