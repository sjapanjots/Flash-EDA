import numpy as np
import pandas as pd
import pytest

import flasheda as fe
from flasheda.analyzers.text import personal_data


@pytest.fixture(scope="module")
def tickets():
    rng = np.random.default_rng(0)
    n = 3_000
    return pd.DataFrame({
        "ticket_id": np.arange(n),
        "message": rng.choice([
            "The app crashes when I open the payment page",
            "Please refund my last order, the item arrived broken",
            "Delivery was late again this week",
            "THANK YOU FOR THE QUICK HELP",
            "",
        ], n),
        "email": [f"customer{i}@example.com" for i in range(n)],
        "channel": rng.choice(["chat", "email", "phone"], n),
        "paid_on": pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
        "amount": rng.lognormal(4, 1, n).round(2).astype(str),
    })


def test_personal_data_kinds():
    s = pd.Series
    assert personal_data(s(["write to a.b@x.io please"] * 10)) == {"email": 100.0}
    assert personal_data(s(["call +91 98765 43210"] * 10)) == {"phone": 100.0}
    assert personal_data(s(["see https://x.io/a"] * 10)) == {"url": 100.0}
    assert personal_data(s(["from 10.0.0.12"] * 10)) == {"ip_address": 100.0}
    assert personal_data(s(["card 4111 1111 1111 1111"] * 10)) == {"card_number": 100.0}


@pytest.mark.parametrize("value", [
    "paid on 2024-01-05 for 1299.50",     # dates and amounts are not phones
    "2024-01-05 12:30",
    "05/01/2024 09:15:00",
    "order 1234567890123",                # 13 digits, fails the Luhn check
    "version 1.2.3.4.5",
    "",
])
def test_no_false_positives(value):
    assert personal_data(pd.Series([value] * 10)) == {}


def test_threshold():
    values = pd.Series(["x@y.com"] + ["nothing"] * 199)        # 0.5% < 1%
    assert personal_data(values) == {}


def test_report(tickets):
    r = fe.analyze(tickets, save_pdf=False)
    t = r.text
    assert t["personal_data_columns"] == ["email"]
    assert t["free_text_columns"] == ["message"]
    ft = t["columns"]["message"]["free_text"]
    assert ft["empty_pct"] == pytest.approx(20, abs=3)
    assert ft["all_caps_pct"] == pytest.approx(20, abs=3)
    vocab = {"app", "crashes", "open", "payment", "page", "please", "refund", "last", "order", "item",
             "arrived", "broken", "delivery", "late", "again", "week", "thank", "quick", "help"}
    assert len(ft["top_words"]) == 10 and set(ft["top_words"]) <= vocab   # no stopwords like "the"
    assert any("personal data in 'email'" in w for w in r.warnings)
    assert "channel" not in t["columns"] and "paid_on" not in t["columns"]   # not free text or PII
    assert r.to_dict()["text"] == t


def test_report_outputs(tickets, tmp_path, capsys):
    r = fe.analyze(tickets, save_pdf=False)
    r.show()
    assert "Free-text columns" in capsys.readouterr().out
    html = r.to_html()
    assert "Text columns" in html and "email 100.0%" in html
    r.save_pdf(str(tmp_path / "r.pdf"))
