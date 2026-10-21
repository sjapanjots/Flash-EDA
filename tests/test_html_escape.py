import pandas as pd

import flasheda as fe

EVIL = "<script>alert(1)</script>"


def test_values_and_names_are_escaped():
    n = 300
    df = pd.DataFrame({
        EVIL: [f"{EVIL}{i % 3}" for i in range(n)],            # name, mode and top values
        "<b>x</b>": range(n),
        "y": [v * 2 for v in range(n)],                         # strong correlation with <b>x</b>
        "contact": [f"user{i}@example.com" for i in range(n)],  # a warning in the banner
    })
    html = fe.analyze(df, target="y").to_html()
    assert "<script>" not in html and "<b>x</b>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html
