<img width="1254" height="1254" alt="FlashEDA" src="https://github.com/user-attachments/assets/a87d02e0-7a18-41e0-844d-a2b9c3d6f769" />

# FlashEDA

[![PyPI](https://img.shields.io/pypi/v/flasheda)](https://pypi.org/project/flasheda/)
[![Python](https://img.shields.io/pypi/pyversions/flasheda)](https://pypi.org/project/flasheda/)
[![Tests](https://github.com/sjapanjots/Flash-EDA/actions/workflows/tests.yml/badge.svg)](https://github.com/sjapanjots/Flash-EDA/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](MIT.License)

**Fixed-time exploratory data analysis for datasets of any size, and validated code to clean, model, check and forecast them.**

FlashEDA analyses a fixed-size random sample, so a 100-million-row file takes about as long as a 100-thousand-row one. It doesn't stop at a report: it writes the cleaning, feature, training, data-check and forecast code for you, as plain Python for **pandas, polars, DuckDB or PySpark**, and runs that code on the sample before handing it over to make sure it works.

[Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [PyPI](https://pypi.org/project/flasheda/)

---

## Contents

- [Why FlashEDA](#why-flasheda)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [User guide](#user-guide)
  - [Analysis and reports](#1-analysis-and-reports)
  - [Target analysis and leakage](#2-target-analysis-and-leakage)
  - [Cleaning code](#3-cleaning-code)
  - [Features](#4-features)
  - [Models](#5-models)
  - [Data contracts](#6-data-contracts)
  - [Comparing datasets (drift)](#7-comparing-datasets-drift)
  - [Databases](#8-databases)
  - [Text and personal data](#9-text-and-personal-data)
  - [Time series and forecasting](#10-time-series-and-forecasting)
  - [Notebook export](#11-notebook-export)
- [Engines and scaling](#engines-and-scaling)
- [API reference](#api-reference)
- [Performance](#performance)
- [Limitations](#limitations)
- [Development](#development)
- [License](#license)

---

## Why FlashEDA

Profiling tools tell you what's wrong with a dataset; you still write the fix by hand, usually in pandas, usually on data that doesn't fit in memory. FlashEDA closes that gap:

| Problem | What FlashEDA does |
|---|---|
| Profiling a large file takes minutes or runs out of memory | Analyses a fixed 5,000-row sample, read with DuckDB without loading the file |
| The report lists problems, but the fixes are manual | Generates a cleaning script, with each step commented with *why* it was added |
| Code written for pandas doesn't scale | The same plan is written for pandas, polars (streaming), DuckDB (out-of-core SQL) or PySpark |
| Generated code can't be trusted | Every script is run on the sample and compared with a reference result before you get it |
| A model looks great because of a leaky column | Per-feature predictive power flags columns that give the answer away |
| New data silently breaks a pipeline | A data contract, learned from good data, checks every new file and can fail a CI job |

## How it works

```
source ──► fixed-size sample ──► analysers ──► report (console / HTML / PDF / JSON)
 (file,      (DuckDB reservoir,     │
  DataFrame,  sampling in the       ▼
  database)   database for SQL)   plan of engine-neutral steps ──► writer ──► script
                                    │                              (pandas, polars,
                                    ▼                               DuckDB, PySpark)
                                  validate: run the script on the sample and
                                  compare with the reference implementation
```

1. **Sample.** The true row count comes from file metadata or `count(*)`; then a uniform random sample (5,000 rows by default) is drawn without loading the full data.
2. **Analyse.** Overview, missing values, numeric and categorical profiles, correlations, text and personal data, and (with a target) task type, class balance, predictive power and leakage.
3. **Plan.** Findings become a list of engine-neutral steps (drop, fill, convert, encode, …), each with its reason and with parameters learned from the sample.
4. **Write.** A writer turns the plan into a standalone script for the chosen engine.
5. **Validate.** The script runs on the sample, and its output is compared with a pandas reference implementation of the same plan.

## Installation

```bash
pip install flasheda
```

Optional extras add engines and features:

| Extra | Adds | Needed for |
|---|---|---|
| `flasheda[fast]` | polars, DuckDB, pyarrow | fast sampling of big files; polars and DuckDB code |
| `flasheda[ml]` | scikit-learn, pyarrow | `flasheda.train` |
| `flasheda[sql]` | SQLAlchemy, pyarrow | database tables and queries |
| `flasheda[spark]` | PySpark | running generated PySpark code |
| `flasheda[all]` | everything above except PySpark | |

```bash
pip install "flasheda[all]"
```

Requires Python 3.9 or newer and pandas 2.0 or newer.

## Quick start

```python
import flasheda as fe

# Analyse any size in seconds
report = fe.analyze("orders.csv", target="churn")
report.show()                         # console report
report.save_html("report.html")       # or report.save_pdf("report.pdf")

# Cleaning code for the full data, already validated on the sample
fe.clean("orders.csv", target="churn", engine="polars").save("clean_orders.py")

# Compare models and get preparation + training scripts
result = fe.train("orders.csv", target="churn")
result.show()
result.save("prep.py", "train.py")    # python prep.py && python train.py

# Guard new data
contract = report.contract()
fe.check("orders_next_month.csv", contract).show()

# Everything above as one runnable notebook
report.to_notebook("orders.ipynb", train=True)
```

## User guide

### 1. Analysis and reports

```python
report = fe.analyze(source, n=None, target=None, engine=None, random_state=42)
```

`source` can be a pandas or polars DataFrame, a CSV/TSV or Parquet path, or a database source from [`fe.sql`](#8-databases).

The report covers:

| Section | Contents |
|---|---|
| Overview | true row count, columns, types, memory (sample and estimated full size), duplicate rows |
| Missing values | null count and % per column, columns above 20% (warn) and 50% (critical), rows with any null |
| Numeric columns | mean, std, min, quartiles, max, IQR outliers, skewness, kurtosis, constant columns |
| Categorical columns | distinct values, cardinality, mode, top values, high-cardinality and likely-ID flags |
| Correlations | Pearson (numbers), Cramér's V (categories), point-biserial (number vs. true/false) |
| Text | personal data and free-text statistics (see [Text](#9-text-and-personal-data)) |
| Target | task, balance, predictive power, leakage, recommended metric and split (see [Target](#2-target-analysis-and-leakage)) |

Short warnings summarise what needs attention, e.g. *"Possible target leakage: refund_issued"*.

```python
report.show()                  # rich console output
report.save_html("r.html")     # standalone HTML page
report.save_pdf("r.pdf")       # PDF
report.to_dict(); report.to_json()
report.warnings                # list of strings
```

Nothing is written to disk unless you ask; `fe.analyze(..., save_pdf=True)` also writes `flasheda_report.pdf`.

The sample size is 5,000 rows by default; change it with `fe.set_sample_size(20_000)` or per call with `n=`.

### 2. Target analysis and leakage

Pass the column you want to predict:

```python
report = fe.analyze("customers.parquet", target="churn")
report.target["task"]                 # "binary" | "multiclass" | "regression"
report.target["feature_power"]        # [{"column": ..., "score": 0..1}, ...] best first
report.target["leakage_columns"]      # columns that predict the target almost perfectly
report.target["recommended_metric"]   # e.g. "PR-AUC / average precision (minority class is 12%...)"
report.target["recommended_split"]    # e.g. "time-based split on signup_date: ..."
```

- **Predictive power** is measured per feature on a held-out part of the sample: a Brier skill score for classification and R² (binned or linear) for regression. ID-like columns score about 0 instead of looking perfect.
- **Leakage**: a feature with power ≥ 0.8 on its own is flagged. It's usually recorded after the outcome (a refund flag for churn, for example).
- **Metric**: PR-AUC for imbalanced binary targets, ROC-AUC for balanced ones, macro-F1 for imbalanced multiclass, MAE for skewed regression targets, otherwise RMSE.
- **Split**: time-based when a date column exists, otherwise stratified (classification) or plain K-fold.

### 3. Cleaning code

```python
code = fe.clean("orders.csv", target="churn", engine="polars", output="orders_clean.parquet")
code.save("clean_orders.py")          # python clean_orders.py → orders_clean.parquet
code.plan.describe()                  # the steps, with reasons
code.validate()                       # [OK] polars: matches reference on 5,000 sampled rows
```

`fe.clean` is a shortcut for `fe.analyze(...).codegen(...)`. The generated script is plain Python with no dependency on FlashEDA. For example:

```python
"""
Cleaning pipeline generated by FlashEDA — engine: polars
Steps:
  1. drop columns ['order_id'] — order_id: numeric ID column
  2. convert 'price' from text to numbers — 97% of values are numbers stored as text
  3. trim and lower-case 'city' — 5 spellings collapse to 2 after trim/lower-case
  4. drop duplicate rows — 0.60% duplicate rows in sample
  5. fill nulls in 'price' with 20.01 — median of sample
  6. log transform 'price' — skewness 1.7076
"""
import polars as pl

def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
    # order_id: numeric ID column
    lf = lf.drop(['order_id'])
    lf = lf.with_columns(
        pl.col('price').cast(pl.String).str.strip_chars(" ").cast(pl.Float64, strict=False),
        pl.col('city').str.strip_chars(" ").str.to_lowercase(),
    )
    lf = lf.unique(maintain_order=True)
    ...

def main() -> None:
    # Lazy scan + streaming sink: the full file is never held in memory.
    transform(pl.scan_csv(SOURCE)).sink_parquet(OUTPUT)
```

**Steps the planner can add**, in this order:

| Step | When |
|---|---|
| Drop columns | likely IDs, more than 50% missing, constant, possible target leakage |
| Drop rows with a null target | the target has nulls |
| Text → numbers | ≥ 95% of values parse as numbers (`"12.5"`, `" 3 "`, with junk like `"N/A"`) |
| Trim + lower-case | spelling variants collapse (`"Delhi"`, `" delhi"`, `"DELHI"`) |
| Drop duplicate rows | duplicates in the sample |
| Group rare values | categories under 1% each become `other` |
| `<col>_was_missing` flag | the target behaves differently when the value is missing |
| Fill nulls | median for numbers, mode for text |
| Log transform | skewed non-negative columns (sign-preserving `log1p`, safe for unseen negatives) |
| Clip outliers | more than 5% IQR outliers |

The target is never transformed. Any phase can be switched off:

```python
report.codegen("duckdb", group_rare=False, clip_outliers=False, drop_leaks=False)
```

Options: `drop_ids`, `drop_missing`, `drop_leaks`, `drop_null_target`, `fix_numeric_text`, `normalize_text`, `drop_duplicates`, `group_rare`, `missing_indicators`, `fill_nulls`, `fix_skew`, `clip_outliers`.

### 4. Features

```python
code = fe.clean("orders.csv", target="churn", features=True)
# or: report.codegen(engine, features=True); report.plan(features={"ratios": False})
```

| Feature | Rule |
|---|---|
| Date parts | ISO date/timestamp text or datetime columns → year, month, day, day of week (ISO, Monday = 1), hour (when times vary) |
| One-hot | text columns with ≤ 15 values |
| Frequency encoding | text columns with ≤ 500 values (each value → its share of the sample) |
| Text length / words | free-text columns → `<col>_length`, `<col>_words` |
| Ratios | `a / b` for the most predictive numeric pairs, kept only if the ratio beats both inputs |
| Dropped | free text after its stats, columns with personal data, text with > 500 values |

### 5. Models

```python
result = fe.train("orders.csv", target="churn", engine="polars",
                  output="features.parquet", model_path="model.joblib")
result.show()                         # leaderboard
result.best                           # e.g. "hist_gradient_boosting"
result.save("prep.py", "train.py")
```

- Compares a baseline, a linear model (logistic regression or ridge), a random forest and histogram gradient boosting on the sample, using the recommended metric and split.
- `prep.py` cleans the full data and adds features in the chosen engine, then writes `features.parquet`.
- `train.py` cross-validates the best model on that file with the same metric and split, fits on all rows (a random 2,000,000 when there are more) and saves `model.joblib`. It's validated by running it on the sample.
- Restrict the candidates with `models=["baseline", "logistic_regression"]`.

### 6. Data contracts

A contract describes what good data looks like. It's learned from a sample, with tolerances, so normal variation passes and real breakage fails.

```python
contract = fe.analyze("orders_jan.parquet").contract()
contract.save("orders.contract.json")

result = fe.check("orders_feb.parquet", "orders.contract.json")   # checks every row
result.ok                                                        # False if any rule fails
result.show()                                                    # the failing rules
```

| Rule | Learned as |
|---|---|
| Type | number, integer, bool, datetime or text, with an allowance for unparseable values |
| Missing | observed % plus a margin that covers the sample's own uncertainty |
| Range | 1st and 99th percentiles stay within bounds (catches unit changes such as ×100) |
| No negatives | for columns never negative in the sample |
| Allowed values | text columns with ≤ 20 values; up to 1% other values |
| Unique | ID columns |

`fe.check` reads files in one DuckDB pass, so it handles files larger than memory. For pipelines and CI:

```python
open("check_orders.py", "w").write(contract.script())
```

```bash
python check_orders.py data/orders_2026_03.parquet   # prints each check, exits 1 on failure
```

The script needs only DuckDB. `contract.to_pandera()` exports the same rules as a [pandera](https://pandera.readthedocs.io) schema.

### 7. Comparing datasets (drift)

```python
drift = fe.compare("orders_jan.parquet", "orders_feb.parquet")
drift.show()
drift.ok                 # False on schema changes or major drift
drift["amount"]          # per-column details
```

- Columns added and removed, and type changes.
- Per column: **PSI** (< 0.1 stable, 0.1–0.25 moderate, ≥ 0.25 major); for numbers also a KS test and the median shift; for text, new categories with examples. A jump in nulls also counts as drift.
- Dates and IDs are reported as range shifts, since they always move forward.

### 8. Databases

```python
src = fe.sql("postgresql://user:pass@host/shop", table="orders")
# or fe.sql(url, query="SELECT * FROM orders WHERE country = 'IN'")

fe.analyze(src, target="churn")
fe.compare(src, "last_month.parquet")
fe.check(src, contract)
src.to_parquet("orders.parquet")      # streams the table in chunks
```

- Works with any SQLAlchemy URL.
- Sampling runs inside the database, so only the sample is transferred: `TABLESAMPLE` on PostgreSQL, `USING SAMPLE` on DuckDB, `ORDER BY random() LIMIT n` elsewhere.
- Passwords are masked in reports (`user:***@host`). Generated scripts read the URL from `$DATABASE_URL`.
- Cleaning code for databases is pandas. For other engines, export with `to_parquet()` first.

### 9. Text and personal data

`fe.analyze` profiles text columns automatically:

- **Personal data.** Emails, phone numbers, URLs, IP addresses and card numbers (Luhn-checked), as the share of each column's non-empty values, with a warning. Dates, times, amounts and order numbers are excluded.
- **Free text** (≥ 3 words or ≥ 30 characters on average). Length, words, % empty, % duplicates, % all caps, and the most common words.

```python
report.text["personal_data_columns"]   # ["email", "contact_note"]
report.text["columns"]["review"]["free_text"]["top_words"]
```

### 10. Time series and forecasting

A random sample breaks time order, so time series use a different path: the **full** data is aggregated per period (one DuckDB `GROUP BY` for files), which stays small at any size.

```python
ts = fe.timeseries("events.parquet", time="created_at", value="amount", agg="sum", freq=None)
ts.show()
ts.trend            # {"direction": "rising", "change_per_year_pct": 18.1, ...}
ts.seasonality      # weekly / daily / yearly strength
ts.anomalies        # unusual periods
```

- `freq` is `"hour"`, `"day"` or `"week"`, chosen from the time span by default.
- `agg="sum"` suits events (sales, orders) and `agg="mean"` suits measurements. Leave out `value` to count rows.
- The report covers empty periods, trend (% per year), seasonality (after removing a moving average, so a random walk isn't mistaken for a pattern), the busiest weekday or hour, and anomalies (robust z-score > 4).

```python
fc = ts.forecast(horizon=14)
fc.show()                                         # backtest leaderboard
fc.forecast                                       # next 14 periods
open("forecast.py", "w").write(fc.script())       # python forecast.py events.parquet
```

`forecast()` compares naive, seasonal naive, moving average, drift and trend + season in a 3-fold rolling backtest (MAE, sMAPE) and keeps the most accurate. These are the baselines any more complex model has to beat. The forecast script reproduces the same numbers from the raw file.

### 11. Notebook export

```python
report = fe.analyze("orders.csv", target="churn")
report.to_notebook("orders.ipynb", train=True, time="created_at", engine="duckdb")
```

One runnable Jupyter notebook with:
- summary and warnings
- the analysis
- the cleaning (and feature) plan and its code
- *optional:* the model comparison and training code (`train=True`)
- *optional:* a time-series profile and forecast (`time=`)
- a contract check (`contract=False` to skip)

Every code cell is a generated script that was validated on the sample, so the notebook runs top to bottom.

## Engines and scaling

| Rows | Default engine | Why |
|---|---|---|
| < 1M | pandas | simplest; fits in memory |
| 1M – 200M | polars (else DuckDB) | lazy, multi-threaded, streams larger-than-memory files |
| > 200M | PySpark, if installed | distributed |

Override with `fe.analyze(..., engine="duckdb")` or `report.codegen("duckdb")`. `fe.available_engines()` shows what's installed.

- **Sampling.** With DuckDB installed, files are sampled with a seeded reservoir sample in one native scan. Without it, a vectorised random-key pass over chunks gives the same uniform sample. Parquet row counts come from metadata.
- **Consistency.** DuckDB reads CSV values as integers, decimals or text only, so `"yes"`/`"no"` aren't turned into booleans. That way every engine sees the same values.
- **Validation.** For each engine, the generated code's output on the sample is compared row by row (order-insensitive, numbers to 6 significant digits) with the pandas reference implementation.

## API reference

| Function | Returns | Purpose |
|---|---|---|
| `fe.analyze(source, n=None, *, target=None, engine=None, random_state=42, show=False, save_pdf=False, pdf_path=...)` | `EDAReport` | profile a source |
| `fe.clean(source, target=None, engine=None, *, output="cleaned.parquet", features=False, validate=True, **plan_options)` | `GeneratedCode` | cleaning (and feature) script |
| `fe.train(source, target, engine=None, *, output="features.parquet", model_path="model.joblib", models=None, n_splits=5)` | `ModelResult` | model comparison, prep and training scripts |
| `fe.check(source, contract)` | `CheckResult` | check full data against a contract (object or JSON path) |
| `fe.compare(reference, current, *, n=None)` | `DriftReport` | schema changes and drift |
| `fe.timeseries(source, time, value=None, *, agg="sum", freq=None)` | `TimeSeriesReport` | time-series profile |
| `fe.sql(url, table=None, query=None)` | `SqlSource` | database table or query as a source |
| `fe.set_sample_size(n)` / `fe.get_sample_size()` | | default sample size (≥ 100) |
| `fe.available_engines()` | `dict` | installed engines |

| `EDAReport` method | Purpose |
|---|---|
| `show()`, `to_html()`, `save_html(path)`, `save_pdf(path)`, `to_dict()`, `to_json()` | output |
| `plan(features=False, **options)` | engine-neutral cleaning plan |
| `codegen(engine=None, *, output=..., features=False, validate=True, **options)` | `GeneratedCode` with `.code`, `.save()`, `.validate()`, `.run(df)` |
| `train(engine=None, **options)` | `ModelResult` |
| `contract(allow_extra_columns=True)` | `Contract` with `.check()`, `.save()`, `.script()`, `.to_pandera()` |
| `to_notebook(path, *, train=False, time=None, value=None, features=False, contract=True, engine=None, source=None)` | notebook path |

`TimeSeriesReport.forecast(horizon=None, methods=None)` returns a `ForecastResult` with `.leaderboard`, `.forecast`, `.show()` and `.script(source=None, output="forecast.csv")`.

## Performance

Measured on synthetic data with typical problems (missing values, junk text, duplicates, skew):

| Task | Time |
|---|---|
| Analyse a 5.1M-row Parquet file | 0.56 s |
| Analyse, plan, generate and validate cleaning code, 1M-row CSV with a target | 2.5 s |
| Run the generated polars cleaning script on 1M rows | 0.9 s |
| `fe.train()` on a 1M-row CSV (features, 4 models, validation) | 6.7 s |
| Generated `prep.py` + `train.py` on 1M rows | 1.3 s + 20 s |
| Learn a contract from 1M rows / run the check script on 1M rows | 0.8 s / 0.6–0.9 s |
| Compare two 1M-row files | 0.6 s |
| Analyse a 1M-row SQLite table (5,000 rows transferred) | 0.3 s |
| Time-series profile of a 5.3M-row Parquet file | 0.7 s |
| Generated forecast script on 5.3M rows | 0.6 s |
| Notebook with cleaning, features, training, forecast and contract | 5 s |

## Limitations

- **Sample-based decisions.** Fill values, clip bounds, kept categories and contract rules come from the sample. Rare values (much less than 1 in 5,000 rows) may not be seen. Use `n=` for a larger sample.
- **CSV row counts** need one full scan (fast in DuckDB, but not constant time); Parquet counts come from metadata.
- **PySpark output** is generated and syntax-checked, but the test suite doesn't execute it.
- **Training** uses scikit-learn in memory; above 2,000,000 rows the training script uses a random subset.
- **Database sampling** with `ORDER BY random()` (SQLite, MySQL, …) scans the table inside the database and isn't seeded.
- **Personal-data detection** is pattern-based. It finds common formats, not every possible one, and isn't a compliance tool.
- **Time series** support hourly, daily and weekly periods.

## Development

```bash
git clone https://github.com/sjapanjots/Flash-EDA.git
cd Flash-EDA
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the checks every change must pass and how releases work.

## License

MIT. See [MIT.License](MIT.License).
