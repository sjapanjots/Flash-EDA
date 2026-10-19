import pandas as pd
import polars as pl
import pytest

from flasheda import engine
from flasheda.engine import backends


def test_choose_engine_by_size():
    assert engine.choose_engine(10_000) == "pandas"
    assert engine.choose_engine(50_000_000) == "polars"
    assert engine.choose_engine(None) in ("polars", "pyspark")


def test_choose_engine_falls_back_when_libraries_missing(monkeypatch):
    monkeypatch.setattr(backends, "is_available", lambda e: e == "pandas")
    assert engine.choose_engine(50_000_000) == "pandas"


def test_prefer_uninstalled_engine_raises(monkeypatch):
    monkeypatch.setattr(backends, "is_available", lambda e: e != "pyspark")
    with pytest.raises(ImportError, match="flasheda\\[spark\\]"):
        engine.choose_engine(10, prefer="pyspark")


@pytest.mark.parametrize("kind", ["csv", "tsv", "parquet"])
def test_inspect_source_reads_true_shape(files, messy, kind):
    info = engine.inspect_source(files[kind])
    assert info.shape == messy.shape
    assert info.columns == list(messy.columns)


def test_inspect_in_memory(messy):
    assert engine.inspect_source(messy).shape == messy.shape
    assert engine.inspect_source(pl.from_pandas(messy)).shape == messy.shape


@pytest.mark.parametrize("kind", ["csv", "tsv", "parquet"])
@pytest.mark.parametrize("method", ["duckdb", "chunked"])
def test_file_sample_is_exact_and_reproducible(files, messy, kind, method):
    a = engine.sample(files[kind], n=1_000, method=method, random_state=7)
    b = engine.sample(files[kind], n=1_000, method=method, random_state=7)
    assert a.shape == (1_000, messy.shape[1])
    pd.testing.assert_frame_equal(a, b)
    assert pd.api.types.is_float_dtype(a["income"])  # dtypes survive sampling


def test_chunked_sample_is_spread_over_whole_file(files, monkeypatch):
    from flasheda.engine import sampler
    monkeypatch.setattr(sampler, "CHUNK_ROWS", 1_000)
    s = engine.sample(files["parquet"], n=2_000, method="chunked")
    ids = s["user id"].str[1:].astype(int)
    assert ids.min() < 1_000 and ids.max() > 19_000


def test_small_source_returned_whole(messy):
    small = messy.head(50)
    assert len(engine.sample(small, n=1_000)) == 50
    assert len(engine.sample(pl.from_pandas(small), n=1_000)) == 50


def test_sample_size_setting():
    original = engine.get_sample_size()
    try:
        engine.set_sample_size(2_000)
        assert engine.get_sample_size() == 2_000
        with pytest.raises(ValueError):
            engine.set_sample_size(10)
    finally:
        engine.set_sample_size(original)


def test_unsupported_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        engine.sample(p)
