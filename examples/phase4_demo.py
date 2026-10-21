"""
Phase 4 demo: data contracts, drift and databases.

    python examples/phase4_demo.py

1. Learns a contract from January (1M rows) and writes a standalone
   DuckDB check script
2. Checks February (passes) and a broken February (fails, exit code 1)
3. Compares January with the broken February
4. Analyses the same data from a SQLite database
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
N = 1_000_000


def make_month(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "order_id": np.arange(seed * n, (seed + 1) * n),
        "amount": rng.lognormal(3, 0.8, n).round(2),
        "country": rng.choice(["IN", "US", "UK", "DE"], n, p=[0.5, 0.3, 0.1, 0.1]),
        "status": rng.choice(["paid", "refunded", "pending"], n, p=[0.85, 0.05, 0.10]),
        "created_at": (pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 31 * 86400, n), unit="s"))
        .strftime("%Y-%m-%d %H:%M:%S"),
    })


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"{label}: {time.perf_counter() - t0:.2f}s")
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    jan, feb, bad = OUT / "jan.parquet", OUT / "feb.parquet", OUT / "feb_broken.csv"
    if not jan.exists():
        make_month(N, 1).to_parquet(jan, index=False)
        f = make_month(N, 2)
        f.to_parquet(feb, index=False)
        broken = f.copy()
        broken["amount"] *= 100                                  # cents instead of rupees
        broken.loc[: N // 20, "country"] = "FR"                  # a new market
        broken.loc[: N // 5, "status"] = None                    # a broken pipeline step
        broken.to_csv(bad, index=False)

    contract = timed("learn contract (1M rows)", lambda: fe.analyze(jan, save_pdf=False).contract())
    contract.save(OUT / "orders.contract.json")
    script = OUT / "check_orders.py"
    script.write_text(contract.script())
    print("\n".join(f"  {line}" for line in contract.describe()))

    for path in (feb, bad):
        t0 = time.perf_counter()
        res = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
        print(f"\n$ python {script.name} {path.name}   → exit {res.returncode} "
              f"({time.perf_counter() - t0:.2f}s)")
        print("\n".join(f"  {l}" for l in res.stdout.splitlines() if l.startswith("[FAIL]") or "passed" in l))

    drift = timed("\ncompare jan vs broken feb", lambda: fe.compare(jan, bad))
    drift.show()

    import sqlalchemy as sa
    db = OUT / "orders.db"
    if not db.exists():
        pd.read_parquet(jan).to_sql("orders", sa.create_engine(f"sqlite:///{db}"), index=False)
    src = fe.sql(f"sqlite:///{db}", table="orders")
    report = timed("\nanalyse SQLite table (sampled in the database)", lambda: fe.analyze(src, save_pdf=False))
    print(f"  {report.original_shape[0]:,} rows, {report.sample_size:,} transferred; "
          f"contract check on the table: {fe.check(src, contract)}")


if __name__ == "__main__":
    main()
