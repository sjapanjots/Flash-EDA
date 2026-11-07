from unittest import mock

import numpy as np
import pandas as pd
import pytest

import flasheda as fe

SECTIONS = ["overview", "missing", "numeric", "categorical", "correlation", "text"]


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "amount": rng.lognormal(3, 1, 600),
        "city": rng.choice(["Pune", "Delhi", None], 600),
        "y": rng.choice(["yes", "no"], 600),
    })


def fail(section):
    return mock.patch(f"flasheda.analyzers.{section}.analyze", side_effect=RuntimeError("boom"))


@pytest.mark.parametrize("section", SECTIONS)
def test_failed_section_is_reported_and_outputs_still_work(df, section, tmp_path, capsys):
    with fail(section):
        report = fe.analyze(df, target="y")
    assert report.errors == {section: "RuntimeError: boom"}
    assert any(w.startswith(f"The {section} analysis failed") for w in report.warnings)
    assert getattr(report, section) == {}
    report.show()                                   # used to crash with KeyError
    assert f"The {section} analysis failed" in capsys.readouterr().out
    assert "boom" in report.to_html()
    report.save_pdf(str(tmp_path / "r.pdf"))
    assert report.to_dict()["errors"] == {section: "RuntimeError: boom"}
    report.plan()                                   # downstream steps still run


@pytest.mark.parametrize("section", SECTIONS)
def test_strict_mode_raises(df, section):
    with fail(section), pytest.raises(RuntimeError, match=f"The {section} analysis failed: RuntimeError: boom"):
        fe.analyze(df, strict=True)


def test_target_failure_always_raises(df):
    with fail("target"), pytest.raises(RuntimeError, match="The target analysis failed"):
        fe.analyze(df, target="y")


def test_no_errors_normally(df):
    assert fe.analyze(df).errors == {}
