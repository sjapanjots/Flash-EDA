<img width="1254" height="1254" alt="image" src="https://github.com/user-attachments/assets/a87d02e0-7a18-41e0-844d-a2b9c3d6f769" />

# FlashEDA

[![PyPI](https://img.shields.io/pypi/v/flasheda)](https://pypi.org/project/flasheda/)
[![Python](https://img.shields.io/pypi/pyversions/flasheda)](https://pypi.org/project/flasheda/)
[![Tests](https://github.com/sjapanjots/Flash-EDA/actions/workflows/tests.yml/badge.svg)](https://github.com/sjapanjots/Flash-EDA/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](MIT.License)

**Current version: 0.6.0** · [Changelog](CHANGELOG.md)

Analyse any dataset in a few seconds, then get **validated cleaning, feature and training code** that runs on the full data, in the engine that fits its size: pandas, polars, DuckDB or PySpark.

FlashEDA analyses a fixed-size random sample, so analysis time barely changes between 100 thousand and 100 million rows.

## Install

```bash
pip install flasheda            # core
pip install "flasheda[fast]"    # + polars, DuckDB, pyarrow (recommended for big files)
pip install "flasheda[ml]"      # + scikit-learn, for flasheda.train
pip install "flasheda[sql]"     # + SQLAlchemy, for database tables
pip install "flasheda[spark]"   # + PySpark
pip install "flasheda[all]"     # everything except PySpark
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

# 3. Model: compare models on the sample, get prep + training scripts
result = fe.train("orders.csv", target="churn", engine="polars")
result.show()                      # leaderboard
result.save("prep.py", "train.py") # python prep.py && python train.py → model.joblib

# 4. Guard: learn rules from good data, check every new file
contract = fe.analyze("orders_jan.csv").contract()
fe.check("orders_feb.csv", contract).show()
open("check.py", "w").write(contract.script())   # python check.py new.csv → exit 1 on failure

# 5. Compare: what changed between two datasets
fe.compare("orders_jan.csv", "orders_feb.csv").show()

# Databases work like files
src = fe.sql("postgresql://user:pass@host/shop", table="orders")
fe.analyze(src, target="churn")

# 6. Time series: trend, seasonality, forecast
ts = fe.timeseries("orders.parquet", time="created_at", value="amount")
ts.show()
fc = ts.forecast()                                 # backtested baselines
open("forecast.py", "w").write(fc.script())        # python forecast.py → forecast.csv

# 7. Everything as one runnable notebook
report.to_notebook("orders.ipynb", train=True, time="created_at")
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
| **Features** | date parts, one-hot / frequency encoding, ratios that beat their inputs |
| **Models** | baseline, linear, random forest, gradient boosting compared with the right metric and split; training script for the winner |
| **Data contracts** | rules learned from good data; full-data checks; standalone check script for CI; pandera export |
| **Drift** | schema changes and per-column drift (PSI, KS, new categories, null-rate change) between two datasets |
| **Databases** | any SQLAlchemy URL; sampling runs in the database; passwords never stored |
| **Text** | personal data (emails, phones, card numbers…) flagged; free-text stats; length/word features |
| **Time series** | full-data aggregation per hour/day/week; gaps, trend, seasonality, anomalies; backtested forecast + script |
| **Notebook** | analysis, cleaning, training, forecast and contract check as one runnable `.ipynb` |
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

### Features and models
`fe.train()` (or `report.train()`) adds features on top of the cleaning steps, compares models on the sample, and writes two scripts:

- **`prep.py`**: cleaning + features for the full data, in the chosen engine
- **`train.py`**: scikit-learn training with cross-validation, using the same metric and split, that saves `model.joblib`

The metric and split follow the target: PR-AUC for imbalanced binary targets, MAE for skewed regression targets, time-based splits when there's a date column, and so on.

### Data contracts and drift
A **contract** describes what good data looks like, learned from a sample with tolerances so normal variation passes:
type, missing %, 1st/99th percentile range (catches unit changes like ×100), no negatives, allowed values, uniqueness of IDs.
`fe.check()` checks every row of a file in one DuckDB pass; `contract.script()` writes the same check as a standalone script that exits with code 1 on failure, ready for a pipeline or CI job.

`fe.compare(old, new)` reports added/removed columns, type changes and how much each column shifted (PSI: < 0.1 stable, 0.1–0.25 moderate, ≥ 0.25 major). Dates and IDs are shown as range shifts, since they always move.

### Time series and forecasting
A random sample breaks time order, so `fe.timeseries()` first aggregates **every row** per period (one DuckDB `GROUP BY`, about 0.7 s for 5M rows), then reports gaps, trend (% per year), daily/weekly/yearly seasonality, the busiest weekday or hour, and anomalies.
`ts.forecast()` backtests five simple baselines (naive, seasonal naive, moving average, drift, trend + season) and keeps the most accurate. These are the numbers any fancier model has to beat. `fc.script()` writes it as a standalone script.

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
| `fe.train()` on a 1M-row CSV (plan, features, 4 models, checks) | 6.7 s |
| Generated `prep.py` + `train.py` on 1M rows | 1.3 s + 20 s |
| Learn a contract from 1M rows / run the check script on 1M rows | 0.8 s / 0.6–0.9 s |
| Compare two 1M-row files | 0.6 s |
| Analyse a 1M-row SQLite table (5,000 rows transferred) | 0.3 s |
| Time-series profile of a 5.3M-row Parquet file | 0.7 s |
| Generated forecast script on 5.3M rows | 0.6 s |
| Notebook with cleaning, features, training, forecast and contract | 5 s |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
