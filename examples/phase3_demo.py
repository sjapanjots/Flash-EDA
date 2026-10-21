"""
Phase 3 demo: features + models.

    python examples/phase3_demo.py

1. Writes a 1M-row orders CSV whose target depends on spend per visit
2. flasheda.train(...) cleans, adds features, compares 4 models on a
   5,000-row sample and writes prep + training scripts
3. Runs both scripts on the full file
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


def make_orders(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    visits = rng.integers(1, 50, n).astype(float)
    spend = visits * rng.lognormal(2, 0.4, n)
    per_visit = spend / visits
    when = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 10**6, n)), unit="min")
    vip = np.where(per_visit * rng.lognormal(0, 0.15, n) > np.quantile(per_visit, 0.85), "yes", "no")
    return pd.DataFrame({
        "order_id": np.arange(n),
        "order_time": when.strftime("%Y-%m-%d %H:%M:%S"),
        "spend": spend.round(2),
        "visits": visits,
        "channel": rng.choice(["web", "app", "store"], n),
        "region": rng.choice([f"r{i}" for i in range(40)], n),
        "vip": vip,
    })


def main() -> None:
    OUT.mkdir(exist_ok=True)
    src = OUT / "orders_1m.csv"
    if not src.exists():
        make_orders(N).to_csv(src, index=False)
    print(f"Source: {src.name} ({src.stat().st_size / 1e6:.0f} MB, {N:,} rows)\n")

    t0 = time.perf_counter()
    res = fe.train(src, target="vip", engine="polars",
                   output=str(OUT / "orders_features.parquet"),
                   model_path=str(OUT / "orders_model.joblib"))
    print(f"analyse + plan + features + compare models + validate: {time.perf_counter() - t0:.1f}s\n")
    res.show()

    print("Feature steps:")
    for line in res.prep.plan.describe():
        if line.startswith(("add ", "one-hot", "frequency")):
            print(f"  - {line}")

    paths = res.save(OUT / "prep_orders.py", OUT / "train_orders.py")
    for name in ("prep", "train"):
        t0 = time.perf_counter()
        print(f"\nRunning {paths[name].name} on the full data...")
        runpy.run_path(str(paths[name]), run_name="__main__")
        print(f"  done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
