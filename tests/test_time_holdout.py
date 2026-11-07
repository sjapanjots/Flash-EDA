import runpy
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import flasheda as fe
import flasheda.modeling as modeling
from flasheda.modeling.specs import candidates

FAST = ["baseline", "ridge"]


def frame(n=2000, seed=0, time=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": x + rng.normal(size=n)})
    if time:
        df["created_at"] = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    return df.sample(frac=1, random_state=seed)        # rows out of time order


def test_latest_rows_are_held_out_of_model_selection():
    seen = []
    real = modeling._evaluate

    def spy(spec, X, y, scoring, cv):
        seen.append(len(X))
        return real(spec, X, y, scoring, cv)

    with mock.patch.object(modeling, "_evaluate", side_effect=spy):
        res = fe.train(frame(), target="y", models=FAST)
    assert res.holdout_rows == 400
    assert set(seen) == {1600}                      # CV never sees the holdout
    assert "holdout" in res.leaderboard.columns
    best = res.leaderboard[res.leaderboard["model"] == res.best].iloc[0]
    assert best["holdout"] < 1.2                    # RMSE close to the noise level (1.0)
    assert "HOLDOUT = 0.2" in res.train_code


def test_training_script_reports_holdout(tmp_path, capsys):
    frame().to_csv(tmp_path / "d.csv", index=False)
    res = fe.train(tmp_path / "d.csv", target="y", models=FAST,
                   output=str(tmp_path / "f.parquet"), model_path=str(tmp_path / "m.joblib"))
    prep, train = res.save(tmp_path / "prep.py", tmp_path / "train.py").values()
    runpy.run_path(str(prep), run_name="__main__")
    runpy.run_path(str(train), run_name="__main__")
    out = capsys.readouterr().out
    assert "holdout RMSE" in out and "latest 400 rows" in out
    assert (tmp_path / "m.joblib").exists()


def test_no_time_column_means_no_holdout():
    res = fe.train(frame(time=False), target="y", models=FAST)
    assert res.holdout_rows == 0
    assert "holdout" not in res.leaderboard.columns
    assert "HOLDOUT = 0.0" in res.train_code


def test_holdout_can_be_turned_off_and_is_validated():
    assert fe.train(frame(), target="y", models=FAST, holdout=0).holdout_rows == 0
    with pytest.raises(ValueError, match="holdout"):
        fe.train(frame(), target="y", models=FAST, holdout=0.7)


def test_random_state_reaches_cv_models_and_script():
    res = fe.train(frame(time=False), target="y", models=["random_forest"], random_state=7)
    assert "RANDOM_STATE = 7" in res.train_code
    assert "random_state=7" in res.train_code
    assert all(s.params.get("random_state", 7) == 7 for s in candidates("binary", 7))
    cv, _ = modeling._make_cv("regression", pd.Series(range(10), dtype=float), 2, False, 7)
    assert cv.random_state == 7
