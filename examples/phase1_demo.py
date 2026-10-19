"""
Phase 1 demo: size-aware engine + validated code generation.

    python examples/phase1_demo.py

1. Writes synthetic "messy" datasets of growing size (Parquet + CSV)
2. Runs flasheda.analyze() on each — time stays roughly flat
3. Generates a cleaning script for every engine and validates it
4. Runs the generated polars script on the largest file
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
SIZES = [100_000, 1_000_000, 5_000_000]


def make_messy(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "order_id": np.char.add("o", np.arange(n).astype(str)),
        "age": np.where(rng.random(n) < 0.1, np.nan, rng.integers(18, 80, n)),
        "income": rng.lognormal(10, 1, n),
        "delivery_days": np.r_[rng.normal(4, 1, n - n // 10), rng.normal(40, 5, n // 10)],
        "city": np.where(rng.random(n) < 0.05, None, rng.choice(["Delhi", "Pune", "Mumbai"], n)),
        "coupon": np.where(rng.random(n) < 0.8, np.nan, rng.random(n)),
        "is_prime": rng.random(n) < 0.3,
    })
    return pd.concat([df, df.iloc[: n // 50]], ignore_index=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("Engines installed:", fe.available_engines(), "\n")

    print(f"{'rows':>12} {'format':>8} {'file MB':>8} {'analyze s':>10}  recommended engine")
    reports = {}
    for n in SIZES:
        df = make_messy(n)
        for fmt in ("parquet", "csv"):
            path = OUT / f"orders_{n}.{fmt}"
            if not path.exists():
                df.to_parquet(path, index=False) if fmt == "parquet" else df.to_csv(path, index=False)
            t0 = time.perf_counter()
            r = fe.analyze(path, save_pdf=False)
            dt = time.perf_counter() - t0
            reports[(n, fmt)] = r
            mb = path.stat().st_size / 1e6
            print(f"{r.original_shape[0]:>12,} {fmt:>8} {mb:>8.1f} {dt:>10.2f}  {r.recommended_engine}")

    big = reports[(SIZES[-1], "parquet")]
    print("\nCleaning plan for the largest dataset:")
    for i, line in enumerate(big.plan().describe(), 1):
        print(f"  {i}. {line}")

    print("\nGenerated code (validated on the sample):")
    for engine in fe.engine.ENGINES:
        code = big.codegen(engine, output=str(OUT / f"cleaned_{engine}.parquet"), validate=False)
        path = code.save(OUT / f"clean_{engine}.py")
        print(f"  {path.relative_to(ROOT.parent)}  {code.validate()}")

    print("\nRunning the generated polars script on the full file...")
    t0 = time.perf_counter()
    runpy.run_path(str(OUT / "clean_polars.py"), run_name="__main__")
    cleaned = pd.read_parquet(OUT / "cleaned_polars.parquet")
    print(f"  {big.original_shape[0]:,} rows in -> {len(cleaned):,} rows out "
          f"in {time.perf_counter() - t0:.2f}s, columns: {list(cleaned.columns)}")


if __name__ == "__main__":
    main()
