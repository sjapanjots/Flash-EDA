# Contributing to FlashEDA

Thanks for helping. This page covers setup, the checks every change must
pass, and how pull requests are organised.

## Setup

```bash
git clone https://github.com/sjapanjots/Flash-EDA.git
cd Flash-EDA
pip install -e ".[dev]"
pytest
```

`[dev]` installs every optional engine (polars, DuckDB, pyarrow,
scikit-learn, SQLAlchemy) plus pandera and nbformat, so the whole test
suite runs. PySpark is not required; PySpark output is checked for valid
syntax.

## Before opening a pull request

- `pytest` passes.
- The code runs on **Python 3.9**: no 3.10+ syntax (`X | Y` types need
  `from __future__ import annotations`), and no backslashes inside f-string
  expressions. CI runs Python 3.9 to 3.13.
- New cleaning or feature steps have a reference implementation in
  `Plan.apply()` and a writer for every engine (pandas, polars, DuckDB,
  PySpark), and a test that validates the generated code against
  `Plan.apply()` for pandas, polars and DuckDB.
- Behaviour that depends on sampled data is tested at realistic sizes, not
  only on tiny frames: rules learned from a 5,000-row sample must hold on
  the full data.

## Pull requests

- **One pull request per change**: a module, a feature, a fix, docs, or a
  release. Don't bundle unrelated changes.
- Each pull request passes the test suite on its own.
- **Every pull request targets `main`.** If it depends on another open pull
  request, branch from that one but keep the base as `main`, and start the
  description with "Merge after #N". Its diff shrinks to its own change once
  #N is merged.
- Version bumps, `CHANGELOG.md` and `README.md` updates go in their own pull
  request at the end.
- Descriptions are short: **What**, **Testing**, and how to release if
  relevant.

## Releasing

1. Bump the version in `setup.py` and `flasheda/__init__.py`, and add a
   section to `CHANGELOG.md` (in its own pull request).
2. After it's merged, create a GitHub release named after the tag, e.g.
   `v0.7.0`. The `Publish to PyPI` workflow checks that the tag matches the
   package version, builds, and uploads using trusted publishing.
