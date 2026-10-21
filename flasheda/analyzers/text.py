"""
flasheda.analyzers.text
~~~~~~~~~~~~~~~~~~~~~~~
Text columns:

  personal data  emails, phone numbers, URLs, IP addresses and card numbers
                 (Luhn-checked) found in any text column, as a share of the
                 column's non-empty values
  free text      for columns that read like sentences (≥ 3 words on average
                 or ≥ 30 characters): length and word statistics, empty and
                 duplicate values, all-caps share, most common words

Regex-based and dependency-free; runs on the fixed sample.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

import pandas as pd

from ..clean.detectors import is_text

PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w.])",
    "url": r"\bhttps?://[^\s]+|\bwww\.[^\s]+",
    "ip_address": r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])",
    "card_number": r"\b(?:\d[ -]?){13,19}\b",
}
PII_MIN_PCT = 1.0                 # report a type when at least 1% of values contain it
FREE_TEXT_MIN_WORDS = 3.0
FREE_TEXT_MIN_CHARS = 30.0
TOP_WORDS = 10
STOPWORDS = frozenset("""a an and are as at be but by for from has have i in is it its of on or
that the this to was were will with you your we our they their he she not no so if when what how
my me all just can do did been than then there about up out very""".split())
WORD = re.compile(r"[a-z][a-z']+")
DATE_LIKE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d:\d{2}")


def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    columns: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        if not is_text(s):
            continue
        values = s.dropna().astype(str)
        info: Dict[str, Any] = {}
        pii = personal_data(values)
        if pii:
            info["personal_data"] = pii
        words = values.str.count(r"\S+")
        lengths = values.str.len()
        if words.mean() >= FREE_TEXT_MIN_WORDS or lengths.mean() >= FREE_TEXT_MIN_CHARS:
            info["free_text"] = _free_text_stats(values, lengths, words)
        if info:
            columns[str(col)] = info

    return {
        "columns": columns,
        "personal_data_columns": [c for c, v in columns.items() if "personal_data" in v],
        "free_text_columns": [c for c, v in columns.items() if "free_text" in v],
    }


def personal_data(values: pd.Series) -> Dict[str, float]:
    """{kind: % of non-empty values containing it}, for kinds in ≥ PII_MIN_PCT of them."""
    if not len(values):
        return {}
    found = {}
    for kind, pattern in PII_PATTERNS.items():
        hits = values.str.contains(pattern, regex=True)
        if kind == "card_number" and hits.any():
            hits = hits & values.map(_has_luhn_number)
        if kind == "phone" and hits.any():
            hits = hits & values.map(_looks_like_phone) & \
                ~values.str.contains(PII_PATTERNS["ip_address"], regex=True)
        pct = round(float(hits.mean() * 100), 2)
        if pct >= PII_MIN_PCT:
            found[kind] = pct
    if "card_number" in found:
        found.pop("phone", None)          # card numbers also match the phone pattern
    return found


def _looks_like_phone(text: str) -> bool:
    """9–15 digits, written with a + or separators, or exactly 10 plain digits.
    Plain digit runs of other lengths are usually order or account numbers."""
    for m in re.finditer(PII_PATTERNS["phone"], text):
        raw = m.group()
        digits = sum(ch.isdigit() for ch in raw)
        formatted = raw.startswith("+") or any(ch in raw for ch in " ().-")
        if DATE_LIKE.search(raw) or re.fullmatch(r"\d+\.\d+", raw):
            continue                       # dates, times and decimals have digit runs too
        if 9 <= digits <= 15 and (formatted or digits == 10):
            return True
    return False


def _has_luhn_number(text: str) -> bool:
    for m in re.finditer(PII_PATTERNS["card_number"], text):
        digits = [int(d) for d in re.sub(r"\D", "", m.group())]
        if 13 <= len(digits) <= 19 and _luhn(digits):
            return True
    return False


def _luhn(digits: List[int]) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


def _free_text_stats(values: pd.Series, lengths: pd.Series, words: pd.Series) -> Dict[str, Any]:
    stripped = values.str.strip()
    letters = stripped.str.replace(r"[^A-Za-z]", "", regex=True)
    counter: Counter = Counter()
    for text in values.str.lower().head(2_000):
        counter.update(w for w in WORD.findall(text) if w not in STOPWORDS)
    n = len(values)
    return {
        "mean_chars": round(float(lengths.mean()), 1),
        "median_chars": float(lengths.median()),
        "max_chars": int(lengths.max()),
        "mean_words": round(float(words.mean()), 1),
        "empty_pct": round(float((stripped == "").mean() * 100), 2),
        "duplicate_pct": round(float(values.duplicated().mean() * 100), 2),
        "all_caps_pct": round(float(((letters.str.len() > 3) & (letters == letters.str.upper())).mean() * 100), 2),
        "top_words": [w for w, _ in counter.most_common(TOP_WORDS)],
        "rows": n,
    }
