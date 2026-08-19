# ⚡ Flash-EDA Package

Constant-time Exploratory Data Analysis — one line, fixed time, any dataset size.

## Install
```bash
pip install flasheda
```

## Usage
```python
import flasheda

report = flasheda.analyze(df)   # works on 1K or 50M rows in the same time
report.show()                   # rich console output
report.save_html("report.html") # browser report
```

## How it works
FlashEDA uses **reservoir sampling** to always analyse exactly 5,000 rows,
regardless of dataset size. All analyzers run in parallel via ThreadPoolExecutor.
