"""
Phase 5 demo: text, time series, forecasting and notebook export.

    python examples/phase5_demo.py

1. 5M support events over 3 years (Parquet): time-series profile of the
   full file, forecast baselines and a standalone forecast script
2. Text profiling of the ticket messages (personal data, free text)
3. One runnable notebook with cleaning, training, forecast and contract
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import flasheda as fe  # noqa: E402

OUT = ROOT / "output"
N_DAYS = 3 * 365


def make_events(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")
    rate = 4_000 * (1 + 0.25 * np.arange(N_DAYS) / 365) * np.where(days.dayofweek >= 5, 0.6, 1.0)
    counts = rng.poisson(rate)
    n = int(counts.sum())
    ts = np.repeat(days.values, counts) + pd.to_timedelta(rng.integers(0, 86_400, n), unit="s").values
    words = np.array("app crash payment refund late delivery login password order broken help".split())
    k = rng.integers(2, 25, n)
    return pd.DataFrame({
        "created_at": ts,
        "message": [" ".join(rng.choice(words, size)) for size in k],
        "email": np.where(rng.random(n) < 0.3, "user@example.com", None),
        "priority": rng.choice(["low", "normal", "high"], n, p=[0.3, 0.6, 0.1]),
        "escalated": np.where(rng.random(n) < 0.05 + 0.4 * (k / 25), "yes", "no"),
    })


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"{label}: {time.perf_counter() - t0:.2f}s")
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    events = OUT / "events.parquet"
    if not events.exists():
        make_events().to_parquet(events, index=False)
    rows = pd.read_parquet(events, columns=["priority"]).shape[0]
    print(f"Source: {events.name} ({events.stat().st_size / 1e6:.0f} MB, {rows:,} rows)\n")

    ts = timed("time-series profile (aggregates every row)", lambda: fe.timeseries(events, time="created_at"))
    ts.show()
    fc = timed("\nforecast backtest", ts.forecast)
    fc.show()
    script = OUT / "forecast_events.py"
    script.write_text(fc.script(output=str(OUT / "forecast.csv")))
    t0 = time.perf_counter()
    subprocess.run([sys.executable, str(script), str(events)], check=True)
    print(f"  forecast script on {rows:,} rows: {time.perf_counter() - t0:.2f}s")

    report = timed("\nanalyze with target", lambda: fe.analyze(events, target="escalated", save_pdf=False))
    for w in report.warnings:
        print(f"  ! {w}")
    ft = report.text["columns"]["message"]["free_text"]
    print(f"  message: {ft['mean_words']} words on average, top words {', '.join(ft['top_words'][:5])}")

    nb = timed("\nnotebook (cleaning + features + training + forecast + contract)",
               lambda: report.to_notebook(OUT / "events.ipynb", train=True, time="created_at",
                                          output=str(OUT / "events_features.parquet")))
    print(f"  wrote {nb}")


if __name__ == "__main__":
    main()
