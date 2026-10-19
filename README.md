<img width="1254" height="1254" alt="image" src="https://github.com/user-attachments/assets/a87d02e0-7a18-41e0-844d-a2b9c3d6f769" />

# FlashEDA

[![PyPI](https://img.shields.io/pypi/v/flasheda)](https://pypi.org/project/flasheda/)
[![Python](https://img.shields.io/pypi/pyversions/flasheda)](https://pypi.org/project/flasheda/)
[![Tests](https://github.com/sjapanjots/Flash-EDA/actions/workflows/tests.yml/badge.svg)](https://github.com/sjapanjots/Flash-EDA/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](MIT.License)

**Current version: 0.3.0** · [Changelog](CHANGELOG.md)

Analyse any dataset in a few seconds, then get **validated cleaning code** that runs on the full data, in the engine that fits its size: pandas, polars, DuckDB or PySpark.

FlashEDA analyses a fixed-size random sample, so analysis time barely changes between 100 thousand and 100 million rows.

## Install

```bash
pip install flasheda            # core
pip install "flasheda[fast]"    # + polars, DuckDB, pyarrow (recommended for big files)
pip install "flasheda[spark]"   # + PySpark
```

## Quick start

```python
import flasheda as fe

# 1. Analyse: any size, a few seconds
report = fe.analyze("orders.csv", target="churn", save_pdf=False)
report.show()                      # console report
report.save_html("report.html")    # or report.save_pdf("report.pdf")

# 2. Clean: get a script for the full data, already checked on the sample
code = fe.clean("orders.csv", target="churn", engine="polars")
code.save("clean_orders.py")       # then: python clean_orders.py
```

## What it does

| | |
|---|---|
| **Overview** | true row count (read from file metadata, no full load), types, memory, duplicates |
| **Data quality** | missing values, outliers, skew, likely ID columns, numbers stored as text, spelling variants, rare categories |
| **Target analysis** | task type, class balance, per-feature predictive power, **possible leakage**, recommended metric and validation split |
| **Correlations** | Pearson (numeric) and Cramér's V (categorical) |
| **Reports** | console, HTML, PDF, JSON |
| **Code generation** | one cleaning plan → pandas, polars (lazy, streaming), DuckDB SQL (larger than memory) or PySpark |
| **Validation** | generated code is run on the sample and compared with a reference result before you get it |

### Cleaning steps
Each step in the generated code has a comment saying why it was added:

- drop likely-ID, mostly-empty, constant and leaky columns
- drop rows where the target is missing, and drop duplicate rows
- convert numbers stored as text (`"12.5"`, `"N/A"`) to numbers
- merge spelling variants (`"Delhi"`, `" delhi"`, `"DELHI"`)
- group rare categories into `other`
- add `<col>_was_missing` flags when missingness predicts the target
- fill nulls (median / mode), log-transform skewed columns, clip outliers

### Engine choice

| Rows | Engine |
|---|---|
| < 1M | pandas |
| 1M – 200M | polars or DuckDB |
| > 200M | PySpark (if installed) |

Override with `fe.analyze(..., engine="duckdb")` or `report.codegen("duckdb")`.

## Performance

Measured on synthetic messy data:

| Task | Time |
|---|---|
| Analyse a 5.1M-row Parquet file | 0.56 s |
| Sample a 1M-row CSV (0.1.x → 0.2.0) | 37.3 s → 0.65 s |
| Analyse + plan + generate + validate, 1M-row CSV with target | 2.5 s |
| Run the generated polars script on 1M rows | 0.9 s |

## Examples

- [`examples/phase1_demo.py`](examples/phase1_demo.py): engine choice and code generation at 0.1M / 1M / 5M rows
- [`examples/phase2_demo.py`](examples/phase2_demo.py): target analysis and cleaning on a 1M-row churn CSV

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
