import ast

import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.codegen import FillNull, Plan, TextStats, generate

RUNNABLE = ["pandas", "polars", "duckdb"]


@pytest.fixture(scope="module")
def support():
    """Longer messages tend to come from unhappy customers (a noisy signal)."""
    rng = np.random.default_rng(1)
    n = 4_000
    words = np.array("order late refund broken app crash payment help please delivery item".split())
    k = rng.integers(1, 30, n)
    message = [" ".join(rng.choice(words, size)) for size in k]
    unhappy = rng.random(n) < 0.1 + 0.6 * (k / 30)
    return pd.DataFrame({
        "message": message,
        "email": [f"user{i % 900}@example.com" for i in range(n)],   # repeats: not an ID
        "channel": rng.choice(["chat", "email", "phone"], n),
        "unhappy": np.where(unhappy, "yes", "no"),
    })


@pytest.fixture(scope="module")
def report(support):
    return fe.analyze(support, target="unhappy", save_pdf=False)


def test_text_stats_semantics():
    df = pd.DataFrame({"t": ["hello  world", "  a b  c ", "", None, "naïve café 😀 ok"]})
    out = Plan([TextStats("t", "t_length", "t_words")]).apply(df)
    assert out["t_length"].tolist()[:3] == [12, 9, 0] and out["t_length"].iloc[4] == 15
    assert out["t_words"].tolist()[:3] == [2, 3, 0] and out["t_words"].iloc[4] == 4
    assert out["t_length"].isna().iloc[3]


@pytest.mark.parametrize("eng", RUNNABLE)
def test_text_stats_all_engines(eng):
    df = pd.DataFrame({"t": ["hello  world", "  a b  c ", "", None, "naïve café 😀 ok", "x"] * 40})
    plan = Plan([TextStats("t", "t_length", "t_words"), FillNull("t_words", 0)])
    result = generate(plan, eng, sample=df).validate()
    assert result.ok and not result.skipped, str(result)


def test_pyspark_text_stats_is_valid_python():
    ast.parse(generate(Plan([TextStats("t", "a", "b")]), "pyspark", source="x.csv").code)


def test_feature_plan_uses_text_and_drops_personal_data(report):
    steps = report.plan(features=True).steps
    stats = [s for s in steps if isinstance(s, TextStats)]
    assert [(s.column, s.length_name, s.words_name) for s in stats] == [
        ("message", "message_length", "message_words")]
    text = "\n".join(report.plan(features=True).describe())
    assert "email: personal data (email)" in text
    assert "message: free text, replaced by length and word count" in text


@pytest.mark.parametrize("eng", RUNNABLE)
def test_text_feature_code_matches_reference(report, eng):
    result = report.codegen(eng, features=True, validate=False).validate()
    assert result.ok and not result.skipped, str(result)


def test_models_use_text_length(report):
    res = report.train(models=["baseline", "logistic_regression"])
    assert {"message_length", "message_words"} <= set(res.features)
    assert "email" not in res.features
    board = res.leaderboard.set_index("model")["score"]
    assert board["logistic_regression"] > board["baseline"] + 0.1
