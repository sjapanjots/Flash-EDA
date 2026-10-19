"""
Phase 2 demo: target analysis + target-aware cleaning.

    python examples/phase2_demo.py

1. Writes a messy 1M-row churn CSV (leak, ID, text-numbers, spelling
   variants, long-tail categories, informative missing values, null targets)
2. flasheda.clean(..., target="churn") → validated polars script
3. Runs the script on the full file
"""

from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import flasheda as fe  # noqa: E402

OUT = ROOT / "output"
N = 1_000_000


def make_churn(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    churn = rng.random(n) < 0.12
    income = rng.lognormal(10, 0.8, n)
    price = rng.lognormal(3, 0.6, n).round(2).astype(str).astype(object)
    price[rng.random(n) < 0.03] = "N/A"
    weights = 1 / np.arange(1, 81) ** 1.6
    label = np.where(churn, "yes", "no").astype(object)
    label[rng.random(n) < 0.02] = None
    return pd.DataFrame({
        "customer_id": np.arange(n),
        "income": np.where(rng.random(n) < np.where(churn, 0.45, 0.05), np.nan, income),
        "price": price,
        "city": rng.choice(["Delhi", "delhi ", "DELHI", "Pune", "pune", "Mumbai"], n),
        "source": rng.choice([f"ref_{i}" for i in range(80)], n, p=weights / weights.sum()),
        "refund_issued": np.where(rng.random(n) < 0.99, churn, ~churn),
        "signup_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(n), unit="min"),
        "churn": label,
    })


def main() -> None:
    OUT.mkdir(exist_ok=True)
    src = OUT / "churn_1m.csv"
    if not src.exists():
        make_churn(N).to_csv(src, index=False)
    print(f"Source: {src.name}  ({src.stat().st_size / 1e6:.0f} MB, {N:,} rows)\n")

    t0 = time.perf_counter()
    code = fe.clean(src, target="churn", engine="polars", output=str(OUT / "churn_clean.parquet"))
    print(f"analyse + plan + generate + validate: {time.perf_counter() - t0:.2f}s\n")

    t = code.report.target
    print(f"Target: {t['target']} ({t['task']}), classes "
          + ", ".join(f"{c['value']} {c['pct']}%" for c in t["classes"]))
    print(f"Metric: {t['recommended_metric']}")
    print(f"Split:  {t['recommended_split']}")
    print("Top predictive features:",
          ", ".join(f"{p['column']} {p['score']}" for p in t["feature_power"][:4]))
    for w in code.report.warnings:
        print(f"  ! {w}")

    print("\nCleaning plan:")
    for i, line in enumerate(code.plan.describe(), 1):
        print(f"  {i:>2}. {line}")
    print(f"\nValidation: {code.validate()}")

    script = code.save(OUT / "clean_churn_polars.py")
    t0 = time.perf_counter()
    runpy.run_path(str(script), run_name="__main__")
    cleaned = pd.read_parquet(OUT / "churn_clean.parquet")
    print(f"Ran {script.name} on the full file in {time.perf_counter() - t0:.2f}s: "
          f"{len(cleaned):,} rows x {cleaned.shape[1]} columns, "
          f"{int(cleaned.drop(columns='churn').isna().sum().sum())} nulls left")


if __name__ == "__main__":
    main()
