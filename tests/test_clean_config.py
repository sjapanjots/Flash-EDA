import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.codegen.steps import ClipOutliers, DateParts, DropColumns

FAST = ["baseline", "logistic_regression"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 600
    amount = rng.normal(0, 10, n)
    amount[:60] = rng.choice([-1, 1], 60) * rng.uniform(100, 1000, 60)
    return pd.DataFrame({
        "order_id": np.arange(n),
        "amount": amount,
        "day": pd.date_range("2025-01-01", periods=n, freq="D").strftime("%d/%m/%Y"),
        "y": rng.integers(0, 2, n),
    })


def test_defaults_match_the_planner():
    cfg = fe.CleanConfig()
    assert cfg.approve == "confident" and not cfg.clip_outliers and cfg.protect == ()


def test_config_drives_the_plan(data):
    report = fe.analyze(data, target="y", save_pdf=False)
    cfg = fe.CleanConfig(protect=["order_id"], clip_outliers=True, features=True)
    plan = report.plan(config=cfg)
    assert not any("order_id" in s.columns for s in plan.steps if isinstance(s, DropColumns))
    assert any(isinstance(s, ClipOutliers) for s in plan.steps)
    assert any(isinstance(s, DateParts) for s in plan.steps)          # features from config

    # keyword options win over the config
    override = report.plan(config=cfg, clip_outliers=False)
    assert not any(isinstance(s, ClipOutliers) for s in override.steps)


def test_config_flows_through_clean_and_train(data, tmp_path):
    path = tmp_path / "d.csv"
    data.to_csv(path, index=False)
    cfg = fe.CleanConfig(protect=("order_id",))
    code = fe.clean(path, target="y", engine="pandas", config=cfg)
    assert not any("order_id" in s.columns for s in code.plan.steps if isinstance(s, DropColumns))
    res = fe.train(path, target="y", models=FAST, config=cfg,
                   output=str(tmp_path / "f.parquet"), model_path=str(tmp_path / "m.joblib"))
    assert "order_id" in res.features


def test_save_load_and_validation(tmp_path):
    cfg = fe.CleanConfig(protect="id", decimal=",", date_formats={"day": "%d/%m/%Y"})
    assert cfg.protect == ("id",)
    again = fe.CleanConfig.load(cfg.save(tmp_path / "c.json"))
    assert again == cfg
    assert cfg.replace(approve="all").approve == "all"
    with pytest.raises(ValueError, match="approve"):
        fe.CleanConfig(approve="yes")
    with pytest.raises(ValueError, match="decimal"):
        fe.CleanConfig(decimal=";")
    with pytest.raises(ValueError, match="Unknown"):
        fe.CleanConfig.from_dict({"clip": True})
