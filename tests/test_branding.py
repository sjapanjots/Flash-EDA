import base64

import pandas as pd

import flasheda as fe
from flasheda.branding import bolt_points, favicon_data_uri, mark_svg


def test_mark_svg_is_self_contained():
    svg = mark_svg(40, "x")
    assert svg.startswith("<svg") and 'width="40"' in svg
    assert 'id="x-bg"' in svg and 'id="x-bolt"' in svg       # ids scoped per use
    decoded = base64.b64decode(favicon_data_uri().split(",", 1)[1]).decode()
    assert decoded.startswith("<svg")


def test_bolt_fits_inside_the_tile():
    for size in (10, 120):
        pts = bolt_points(size)
        assert all(0 < x < size and 0 < y < size for x, y in pts)


def test_reports_are_branded(tmp_path):
    report = fe.analyze(pd.DataFrame({"a": range(300), "b": ["x", "y", "z"] * 100}))
    html = report.to_html()
    assert '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,' in html
    assert 'aria-label="FlashEDA"' in html
    pdf = report.save_pdf(str(tmp_path / "r.pdf"))
    assert pdf.stat().st_size > 1_000
