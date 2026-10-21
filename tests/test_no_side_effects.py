import os

import pandas as pd

import flasheda as fe


def test_analyze_writes_nothing_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fe.analyze(pd.DataFrame({"a": range(200), "b": ["x", "y"] * 100}))
    assert os.listdir(tmp_path) == []


def test_pdf_on_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fe.analyze(pd.DataFrame({"a": range(200)}), save_pdf=True)
    assert os.listdir(tmp_path) == ["flasheda_report.pdf"]
    fe.analyze(pd.DataFrame({"a": range(200)}), save_pdf=True, pdf_path=str(tmp_path / "x.pdf"))
    assert (tmp_path / "x.pdf").exists()
