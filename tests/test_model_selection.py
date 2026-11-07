import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.modeling import ModelSpec


def frame(signal: bool, n=1_500, seed=0):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    logit = 1.5 * x if signal else np.zeros(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def test_noise_target_warns_that_nothing_beats_baseline():
    res = fe.train(frame(signal=False), target="y", models=["logistic_regression"])
    assert not res.beats_baseline
    assert any("clearly better than the baseline" in w for w in res.warnings)
    assert "baseline" in set(res.leaderboard["model"])        # added automatically


def test_real_signal_beats_baseline():
    res = fe.train(frame(signal=True), target="y", models=["baseline", "logistic_regression"])
    assert res.beats_baseline and res.best == "logistic_regression"
    assert not any("baseline" in w for w in res.warnings)


def test_custom_model_spec(tmp_path):
    knn = ModelSpec("knn", "sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 25}, scale=True)
    tree = ModelSpec("tree", "sklearn.tree", "DecisionTreeClassifier",
                     {"max_depth": 3, "random_state": 0})
    res = fe.train(frame(signal=True), target="y", models=[knn, tree, "logistic_regression"],
                   random_state=5, output=str(tmp_path / "f.parquet"), model_path=str(tmp_path / "m.joblib"))
    assert {"baseline", "knn", "tree", "logistic_regression"} == set(res.leaderboard["model"])
    custom = fe.train(frame(signal=True), target="y", models=[tree], random_state=5)
    assert custom.best == "tree"
    assert "DecisionTreeClassifier(max_depth=3, random_state=5)" in custom.train_code   # seeded


def test_bad_model_lists():
    with pytest.raises(ValueError, match="Unknown model 'svm'"):
        fe.train(frame(True), target="y", models=["svm"])
    twice = ModelSpec("logistic_regression", "sklearn.linear_model", "LogisticRegression")
    with pytest.raises(ValueError, match="unique"):
        fe.train(frame(True), target="y", models=["logistic_regression", twice])
