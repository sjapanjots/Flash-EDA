import numpy as np
import pandas as pd

import flasheda as fe


def data():
    rng = np.random.default_rng(0)
    amount = rng.normal(0, 10, 3000)                  # e.g. balance changes: can be negative
    amount[:300] = rng.choice([-1, 1], 300) * rng.uniform(100, 1000, 300)   # 10% real, large
    return pd.DataFrame({"amount": amount, "qty": rng.integers(1, 5, 3000)})


def test_outliers_are_flagged_not_changed_by_default():
    report = fe.analyze(data(), save_pdf=False)
    plan = report.plan()
    assert not any("clip" in d for d in plan.describe())
    assert len(plan.suggestions) == 1 and plan.suggestions[0].startswith("amount:")
    assert "clip_outliers=True" in plan.suggestions[0]
    cleaned = plan.apply(report.sample)
    assert cleaned["amount"].max() == report.sample["amount"].max()   # large values kept


def test_suggestions_appear_in_generated_code():
    code = fe.analyze(data(), save_pdf=False).codegen("pandas").code
    assert "Not applied (review these yourself):" in code
    assert "amount: " in code


def test_clipping_is_opt_in():
    report = fe.analyze(data(), save_pdf=False)
    plan = report.plan(clip_outliers=True)
    assert any(d.startswith("clip 'amount'") for d in plan.describe())
    assert plan.suggestions == []
