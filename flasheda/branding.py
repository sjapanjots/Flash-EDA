"""
flasheda.branding
~~~~~~~~~~~~~~~~~
The FlashEDA mark for generated outputs (HTML report, PDF, notebooks).

Outputs use the small-size mark (a solid bolt on the navy tile): it stays
crisp at header and favicon sizes, where the full dotted mark would blur.
The full logo set lives in assets/logo/ in the repository.
"""

from __future__ import annotations

import base64
from typing import List, Tuple

NAVY, NAVY_LIGHT = "#0B1220", "#16233D"
AMBER_LIGHT, AMBER = "#FDE047", "#F59E0B"

LOGO_URL = ("https://raw.githubusercontent.com/sjapanjots/Flash-EDA/main/"
            "assets/logo/flasheda-mark-128.png")

# Slanted lightning bolt, centred in a 120×120 tile (same shape as assets/logo).
_BOLT = [(60, 2), (30, 52), (50, 52), (38, 98), (74, 42), (56, 42), (78, 2)]


def bolt_points(size: float = 120.0, scale: float = 0.92) -> List[Tuple[float, float]]:
    """Bolt vertices for a tile `size` units wide, centred like the logo."""
    xs, ys = [p[0] for p in _BOLT], [p[1] for p in _BOLT]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    k = size / 120
    return [(((x - cx) * scale + 60) * k, ((y - cy) * scale + 60) * k) for x, y in _BOLT]


def mark_svg(size: int = 40, uid: str = "fe") -> str:
    """Inline SVG of the small mark; `uid` keeps gradient ids unique on a page."""
    d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in bolt_points()) + " Z"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 120 120" role="img" aria-label="FlashEDA"><defs>'
        f'<linearGradient id="{uid}-bg" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{NAVY_LIGHT}"/><stop offset="1" stop-color="{NAVY}"/></linearGradient>'
        f'<linearGradient id="{uid}-bolt" x1="0" y1="0" x2="0.6" y2="1">'
        f'<stop offset="0" stop-color="{AMBER_LIGHT}"/><stop offset="1" stop-color="{AMBER}"/></linearGradient>'
        f'</defs><rect width="120" height="120" rx="28" fill="url(#{uid}-bg)"/>'
        f'<path d="{d}" fill="url(#{uid}-bolt)"/></svg>'
    )


def favicon_data_uri() -> str:
    """The mark as a data: URI for <link rel="icon">."""
    return "data:image/svg+xml;base64," + base64.b64encode(mark_svg(64, "fav").encode()).decode()


def draw_pdf_mark(pdf, x: float, y: float, size: float) -> None:
    """Draw the mark with fpdf2 primitives (no image file needed)."""
    pdf.set_fill_color(*_rgb(NAVY_LIGHT))
    pdf.rect(x, y, size, size, style="F", round_corners=True, corner_radius=size * 28 / 120)
    pdf.set_fill_color(*_rgb(AMBER))
    pdf.polygon([(x + px, y + py) for px, py in bolt_points(size)], style="F")


def _rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
