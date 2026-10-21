import json

import pytest

import flasheda as fe
from test_features_model import make_shop


@pytest.fixture(scope="module")
def shop_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("nb") / "shop.csv"
    make_shop(3_000).to_csv(path, index=False)
    return path


def run_notebook(path, workdir, monkeypatch):
    """Execute every code cell in order in one namespace, like a kernel would."""
    monkeypatch.chdir(workdir)
    nb = json.loads(path.read_text())
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"<cell {i}>", "exec"), ns)
    return nb, ns


def test_minimal_notebook(shop_csv, tmp_path, monkeypatch):
    report = fe.analyze(shop_csv, save_pdf=False)
    path = report.to_notebook(tmp_path / "a.ipynb")
    nb, ns = run_notebook(path, tmp_path, monkeypatch)
    titles = [c["source"][0] for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert titles[0].startswith("# FlashEDA analysis: `shop.csv`")
    assert any(t.startswith("## Cleaning") for t in titles)
    assert (tmp_path / "cleaned.parquet").exists() and (tmp_path / "contract.json").exists()
    assert ns["result"].ok


def test_full_notebook(shop_csv, tmp_path, monkeypatch):
    report = fe.analyze(shop_csv, target="vip", save_pdf=False)
    path = report.to_notebook(tmp_path / "full.ipynb", engine="duckdb", train=True,
                              time="order_time", output="features.parquet")
    nb, ns = run_notebook(path, tmp_path, monkeypatch)
    text = "".join("".join(c["source"]) for c in nb["cells"])
    for heading in ("## Warnings", "## Cleaning and features (duckdb)", "## Model", "## Time series",
                    "## Data contract"):
        assert heading in text
    assert (tmp_path / "features.parquet").exists() and (tmp_path / "model.joblib").exists()
    assert ns["fc"].method in ns["fc"].leaderboard["method"].tolist()


def test_valid_nbformat(shop_csv, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    report = fe.analyze(shop_csv, target="vip", save_pdf=False)
    nbformat.validate(nbformat.read(str(report.to_notebook(tmp_path / "v.ipynb", train=True)), as_version=4))


def test_errors(shop_csv):
    df = make_shop(500)
    with pytest.raises(ValueError, match="source="):
        fe.analyze(df, save_pdf=False).to_notebook("x.ipynb")
    with pytest.raises(ValueError, match="needs a target"):
        fe.analyze(shop_csv, save_pdf=False).to_notebook("x.ipynb", train=True)


def test_dataframe_report_with_explicit_source(shop_csv, tmp_path, monkeypatch):
    import pandas as pd
    report = fe.analyze(pd.read_csv(shop_csv), save_pdf=False)
    path = report.to_notebook(tmp_path / "df.ipynb", source=str(shop_csv), contract=False)
    run_notebook(path, tmp_path, monkeypatch)
    assert (tmp_path / "cleaned.parquet").exists()
