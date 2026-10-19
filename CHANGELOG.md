# Changelog

All notable changes to FlashEDA. Versions follow [Semantic Versioning](https://semver.org/).

## [0.3.0] — Phase 2: target analysis and target-aware cleaning

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

## [0.2.0] — Phase 1: size-aware engine and code generation

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

[0.3.0]: https://github.com/sjapanjots/Flash-EDA/releases/tag/v0.3.0
[0.2.0]: https://github.com/sjapanjots/Flash-EDA/compare/41829f5...41c1278
