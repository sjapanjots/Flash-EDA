import runpy

import joblib
import numpy as np
import pandas as pd
import pytest

import flasheda as fe

FAST = ["baseline", "logistic_regression"]


def frame(labels, seed=0, time=False):
    rng = np.random.default_rng(seed)
    n = len(labels)
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "y": labels})
    if time:
        df["created_at"] = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    return df


def test_single_minority_row_explains_instead_of_crashing():
    df = frame(["a"] * 2999 + ["b"])
    with pytest.raises(ValueError, match=r"'b' \(1 row\).*larger sample"):
        fe.train(df, target="y", models=FAST)


def test_few_minority_rows_reduce_folds_and_warn():
    rng = np.random.default_rng(1)
    labels = np.array(["a"] * 2997 + ["b"] * 3)
    rng.shuffle(labels)
    res = fe.train(frame(list(labels)), target="y", models=FAST)
    assert any("3 folds instead of 5" in w for w in res.warnings)
    assert any("only 3 sampled rows" in w for w in res.warnings)


def test_multiclass_rare_class_policies(tmp_path):
    rng = np.random.default_rng(2)
    labels = list(rng.choice(["a", "b", "c"], 2999)) + ["d"]
    df = frame(labels)
    with pytest.raises(ValueError, match="rare_classes='drop' or 'merge'"):
        fe.train(df, target="y", models=FAST)

    dropped = fe.train(df, target="y", models=FAST, rare_classes="drop")
    assert any("Dropped rows of rare target classes: 'd'" in w for w in dropped.warnings)
    assert "isin(['d'])" in dropped.train_code

    with pytest.raises(ValueError, match="Even merged"):
        fe.train(df, target="y", models=FAST, rare_classes="merge")
    two_rare = frame(labels[:-2] + ["d", "e"])
    merged = fe.train(two_rare, target="y", models=FAST, rare_classes="merge",
                      output=str(tmp_path / "f.parquet"), model_path=str(tmp_path / "m.joblib"))
    assert any("Merged rare target classes into 'other'" in w for w in merged.warnings)

    two_rare.to_csv(tmp_path / "d.csv", index=False)
    res = fe.train(tmp_path / "d.csv", target="y", models=FAST, rare_classes="merge",
                   output=str(tmp_path / "f.parquet"), model_path=str(tmp_path / "m.joblib"))
    paths = res.save(tmp_path / "prep.py", tmp_path / "train.py")
    runpy.run_path(str(paths["prep"]), run_name="__main__")
    runpy.run_path(str(paths["train"]), run_name="__main__")
    assert set(joblib.load(tmp_path / "m.joblib").classes_) == {"a", "b", "c", "other"}


def test_bad_policy():
    with pytest.raises(ValueError, match="rare_classes must be"):
        fe.train(frame(["a", "b"] * 200), target="y", rare_classes="ignore")


def test_time_split_needs_every_class_in_every_fold():
    labels = ["b"] * 30 + ["a"] * 2970                  # the rare class only occurs early on
    with pytest.raises(ValueError, match="time-based split"):
        fe.train(frame(labels, time=True), target="y", models=FAST)
