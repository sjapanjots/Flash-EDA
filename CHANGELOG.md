# Changelog

All notable changes to FlashEDA. Versions follow [Semantic Versioning](https://semver.org/).

## [0.7.0] — Safer cleaning and honest evaluation

Fixes from an independent test review.

### Added
- **Risk, confidence and approval** for every cleaning step: a step registry (risk, reversibility, engines), a per-step confidence from the sample, and `approve="confident" | "all" | "safe"`. Skipped steps are kept in `plan.skipped` and listed in the generated script; `plan.review()` shows them all
- `plan.impact(source)`: a dry run on the full data (rows removed, columns dropped/added, nulls and distinct values before/after)
- `protect=[...]`: columns that are never dropped or changed
- Numbers with thousands separators, decimal commas and currency (`"1,250.50"`, `"₹1,25,000"`, `"1.250,50"`, `"12%"`) in all four engines; `decimal=` when the format is ambiguous
- Dates in common non-ISO formats (`25/12/2026`, `7-Nov-2026`); ambiguous day/month order is never guessed, set it with `date_formats=`
- `fe.CleanConfig`: every cleaning choice in one object that saves to JSON
- `fe.sample_size(n)`: a context-local sample size, safe in threads and async code
- `train()`: an untouched holdout of the latest 20% of rows for time-based data (`holdout=`); `rare_classes="error" | "drop" | "merge"`; custom `ModelSpec` candidates; `result.beats_baseline` with a warning when no model beats the baseline
- `analyze(strict=False)`: a failed analysis section is reported in `report.errors` and the warnings instead of disappearing
- CI job that runs the generated PySpark code; acceptance tests with hand-written expected outputs; SQL dialect tests

### Changed
- **Outlier clipping is off by default**: outliers are listed in `plan.suggestions`; pass `clip_outliers=True` to clip as before
- High-risk steps the sample isn't sure about (e.g. a column 51% missing) are skipped until approved; pass `approve="all"` for the old behaviour
- Time columns are found by their values, not only by names like `date` or `_at`
- `random_state` now reaches cross-validation, the models and `train.py` (was fixed at 42)
- The baseline model is always part of the comparison

### Fixed
- A failed analysis section made `report.show()` crash
- `train()` crashed with an unclear scikit-learn error when a target class had 1 or 2 rows; time-based splits could leave a class out of a fold
- Generated scripts could contain `np.str_('yes')` (pandas 2), which fails with a NameError
- Sampling and column listing failed on SQL Server and Oracle (no `LIMIT`; Oracle rejects `AS` on subquery aliases)
- Generated PySpark code crashed on Spark 4 (ANSI mode) when text didn't parse as a number or date; it now returns null as on Spark 3

### Not changed
- The engines keep separate code writers rather than a shared execution layer; acceptance tests and per-engine validation keep them consistent

## [0.6.0] — Text, time series and notebooks

### Added
- **Text profiling** in `analyze()`: personal data (emails, phone numbers, URLs, IP addresses, Luhn-checked card numbers) as a share of each text column's non-empty values, with a warning; for free-text columns, length, word, empty, duplicate and all-caps statistics and top words
- **Text features**: free-text columns become `<col>_length` and `<col>_words` (all four engines); columns with personal data are dropped from model features
- `flasheda.timeseries(source, time=..., value=...)`: aggregates the *full* data per hour/day/week (one DuckDB `GROUP BY` for files) and reports gaps, trend, seasonality, weekday/hour profile and anomalies
- `ts.forecast()`: backtests naive, seasonal naive, moving average, drift and trend + season baselines; `forecast.script()` writes a standalone forecast script that reproduces the result
- `report.to_notebook(path, train=, time=, contract=)`: one runnable Jupyter notebook with the analysis, cleaning/feature code, training, forecast and contract check
- Point-biserial correlation for numeric–boolean pairs (promised by the docs, never computed); strong pairs are reported with a warning

### Changed
- `analyze()` no longer writes `flasheda_report.pdf` by default; pass `save_pdf=True` or call `report.save_pdf()`

### Fixed
- Free text was flagged as a likely ID column (every sentence is unique) and dropped before it could be used
- The HTML report's original tables and warning banners didn't escape column names and values, so text like `<script>` became markup

## [0.5.0] — Data contracts, drift and databases

### Added
- **Data contracts**: `report.contract()` learns per-column rules from the sample with tolerances (type, missing %, percentile range, no negatives, allowed values, uniqueness); `flasheda.check(source, contract)` checks the *full* data (one DuckDB pass for files, pandas for DataFrames); contracts save to JSON
- `contract.script()`: a standalone DuckDB check script that exits with code 1 on failure, for pipelines and CI
- `contract.to_pandera()`: the same rules as a pandera schema
- `flasheda.compare(reference, current)`: added/removed columns, type changes, and per-column drift (PSI, KS test, median shift, new categories, null-rate change); dates and IDs are reported as range shifts
- `flasheda.sql(url, table=... | query=...)`: database tables and queries as sources for analyze, compare, contracts and pandas cleaning code; sampling runs in the database; passwords are masked and generated scripts read `$DATABASE_URL`; `.to_parquet()` streams a table to a file
- Optional install: `flasheda[sql]`

## [0.4.0] — Features and models

### Added
- `report.plan(features=True)` / `codegen(..., features=True)` / `clean(..., features=True)`: model-ready features, generated for all four engines and checked on the sample
  - date/timestamp columns → year, month, day, day of week (ISO), hour
  - categories: ≤ 15 values → one-hot; ≤ 500 → frequency encoding; more → dropped as free text
  - ratios `a / b` between numeric columns, kept only when they predict the target clearly better than both inputs (computed before log/clip transforms)
- `flasheda.train(source, target=...)` / `report.train()`: compares baseline, linear, random-forest and gradient-boosting models on the sample, using the recommended metric and split (time-based when there is a date column); returns a leaderboard and two scripts:
  - `prep.py`: cleaning + features on the full data (any engine)
  - `train.py`: scikit-learn training with cross-validation that saves `model.joblib`; checked by running it on the sample
- Optional installs: `flasheda[ml]` (scikit-learn) and `flasheda[all]`

### Fixed
- Rare-value grouping kept at most 25 values, so common values were lumped into `other` too

## [0.3.0] — Target analysis and target-aware cleaning

### Added
- `analyze(source, target="col")`: target analysis
  - task detection: binary, multiclass or regression
  - class balance and imbalance ratio
  - per-feature predictive power (0–1, measured on a held-out split of the sample)
  - possible-leakage flags for features that predict the target almost perfectly
  - recommended metric and validation split (time-based when a date column exists)
  - target section in the console, HTML and PDF reports
- `flasheda.clean(source, target=..., engine=...)`: one call from raw data to a validated cleaning script
- New cleaning steps in all four engines: drop rows with a null target, numbers stored as text → numbers, trim + lower-case spelling variants, group rare categories into `other`, `<col>_was_missing` flags when missingness predicts the target
- The planner drops numeric ID columns and leaky columns, and never transforms the target

### Fixed
- PDF export crashed on any non-Latin-1 character, including the footer's own "—", so `analyze()` with default settings failed
- DuckDB read `yes`/`no` CSV values as booleans, unlike pandas and polars
- Cramér's V ran on ID-like text columns (5.8 s on a 5,000-row sample)
- Unique timestamps were flagged as ID columns
- `correlation.py` used Python 3.10+ syntax although the package declared 3.8 support

### Changed
- Requires Python ≥ 3.9 and pandas ≥ 2.0

## [0.2.0] — Size-aware engine and code generation

### Added
- `flasheda.engine`: reads the true row count without loading the data (Parquet metadata, DuckDB count for CSV); fast sampling with a DuckDB reservoir or vectorised chunks; picks pandas / polars / DuckDB / PySpark by size
- `flasheda.codegen`: engine-neutral cleaning plan, code writers for pandas, polars (lazy, streaming), DuckDB SQL and PySpark, and a validator that runs the generated code on the sample
- `EDAReport.plan()` and `EDAReport.codegen(engine)`
- Optional installs: `flasheda[fast]`, `flasheda[spark]`, `flasheda[dev]`

### Fixed
- File sources reported the sample size (5,000) as the dataset's row count
- CSV sampling looped over every row in Python: about 57× faster on a 1M-row CSV
- Text columns weren't recognised on pandas 3

## [0.1.3] and earlier
- Fixed-sample EDA: overview, missing values, numeric and categorical summaries, correlations; console, HTML and PDF reports.

[0.7.0]: https://github.com/sjapanjots/Flash-EDA/releases/tag/v0.7.0
[0.6.0]: https://github.com/sjapanjots/Flash-EDA/releases/tag/v0.6.0
[0.5.0]: https://github.com/sjapanjots/Flash-EDA/releases/tag/v0.5.0
[0.4.0]: https://github.com/sjapanjots/Flash-EDA/releases/tag/v0.4.0
[0.3.0]: https://github.com/sjapanjots/Flash-EDA/releases/tag/v0.3.0
[0.2.0]: https://github.com/sjapanjots/Flash-EDA/compare/41829f5...41c1278
