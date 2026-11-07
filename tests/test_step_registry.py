import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.codegen import REGISTRY, StepInfo, generate, info, register
from flasheda.codegen import steps as S
from flasheda.codegen.registry import margin_confidence, proportion_confidence


def test_every_step_type_is_registered():
    for t in S.ROW_STEPS + S.COLUMN_STEPS:
        assert t in REGISTRY
    assert info(S.DropColumns(("a",))).risk == "high"
    assert info(S.Log1p("a")).reversible


def test_confidence_math():
    assert proportion_confidence(0.9, 0.5, 5000) == 1.0
    assert 0.5 <= proportion_confidence(0.51, 0.5, 100) < 0.7
    assert margin_confidence(1.0, 1.0, 0.1) == 0.5


def borderline():
    """'notes' is 51% missing on 200 rows: just over the 50% bar, not a sure thing."""
    rng = np.random.default_rng(0)
    n = 200
    notes = pd.Series(rng.choice(["a", "b", "c"], n), dtype=object)
    notes[:102] = None
    return pd.DataFrame({"x": rng.normal(size=n), "notes": notes,
                         "empty": [None] * 180 + ["z"] * 20})


def test_uncertain_destructive_step_is_skipped_by_default():
    report = fe.analyze(borderline(), save_pdf=False)
    plan = report.plan()
    dropped = [c for s in plan.steps if isinstance(s, S.DropColumns) for c in s.columns]
    assert "empty" in dropped and "notes" not in dropped     # 90% missing is sure, 51% is not
    (step, why), = plan.skipped
    assert step.columns == ("notes",) and step.confidence < 0.95 and "confidence" in why

    review = plan.review()
    row = review[review["status"] == "skipped"].iloc[0]
    assert row["risk"] == "high" and not row["reversible"]

    code = report.codegen("pandas").code
    assert "Skipped, needs your approval" in code and "'notes'" in code


def test_approval_policies():
    report = fe.analyze(borderline(), save_pdf=False)
    everything = report.plan(approve="all")
    assert not everything.skipped
    assert any("notes" in s.columns for s in everything.steps if isinstance(s, S.DropColumns))
    safe = report.plan(approve="safe")
    assert all(info(s).risk == "low" for s in safe.steps)
    assert {type(s) for s, _ in safe.skipped} >= {S.DropColumns}
    with pytest.raises(ValueError, match="approve"):
        report.plan(approve="maybe")


def test_engine_support_is_checked():
    class Custom(S.Log1p):
        pass

    register(Custom, StepInfo("low", True, "test", engines=("pandas",)))
    try:
        with pytest.raises(ValueError, match="polars writer can't generate: Custom"):
            generate(S.Plan([Custom("x")]), "polars")
        with pytest.raises(ValueError, match="risk"):
            register(Custom, StepInfo("extreme", True, "test"))
    finally:
        REGISTRY.pop(Custom, None)
