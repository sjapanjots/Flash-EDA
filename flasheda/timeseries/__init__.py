"""
flasheda.timeseries
~~~~~~~~~~~~~~~~~~~
Time-series profile of a dataset:

    ts = flasheda.timeseries("orders.parquet", time="created_at", value="amount")
    ts.show()

A random sample breaks time order, so this module first aggregates the
*full* data per period (one DuckDB GROUP BY over a file, pandas for a
DataFrame), which is small even for billions of rows, then analyses that
series:

  frequency   hour / day / week, chosen from the time span (or given)
  gaps        periods with no rows, and the longest run of them
  trend       linear fit: rising / falling / flat, % change per year
  seasonality autocorrelation at natural lags (24h, 7d, 52w, …) after
              removing the trend; ≥ 0.3 counts as seasonal
  profile     average by weekday (daily data) or hour of day (hourly data)
  anomalies   periods more than 4 robust standard deviations from the median
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..engine.backends import is_available
from ..engine.source import CSV_EXTS, DUCKDB_CSV_TYPES, PARQUET_EXTS, _is_polars_frame, _is_sql, csv_sep

__all__ = ["TimeSeriesReport", "aggregate", "timeseries"]

FREQS = ("hour", "day", "week")
PERIODS_PER_YEAR = {"hour": 24 * 365.25, "day": 365.25, "week": 52.18}
SEASONAL_LAGS = {"hour": [(24, "daily"), (168, "weekly")],
                 "day": [(7, "weekly"), (365, "yearly")],
                 "week": [(52, "yearly")]}
SEASONAL_MIN_ACF = 0.3
TREND_MAX_P = 0.01
TREND_MIN_CHANGE = 0.05          # total fitted change must be ≥ 5% of the mean
ANOMALY_Z = 4.0


@dataclass
class TimeSeriesReport:
    time: str
    value: Optional[str]
    agg: str
    freq: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_periods: int
    missing_periods: int
    longest_gap: int
    rows_per_period: float
    trend: Dict[str, Any]
    seasonality: List[Dict[str, Any]]
    profile: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    series: pd.DataFrame = field(repr=False)

    @property
    def metric(self) -> str:
        return f"{self.agg}({self.value})" if self.value else "rows"

    @property
    def season_length(self) -> Optional[int]:
        """Lag of the strongest seasonality, if any."""
        found = [s for s in self.seasonality if s["seasonal"]]
        return max(found, key=lambda s: s["strength"])["lag"] if found else None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (str(v) if isinstance(v, pd.Timestamp) else v)
                for k, v in self.__dict__.items() if k != "series"}

    def show(self) -> None:
        from rich.console import Console

        c = Console()
        c.print(f"[bold]Time series[/bold] {self.metric} per {self.freq}, "
                f"{self.start:%Y-%m-%d} → {self.end:%Y-%m-%d} ({self.n_periods:,} periods)")
        if self.missing_periods:
            c.print(f"  [yellow]gaps:[/] {self.missing_periods} empty periods, longest {self.longest_gap}")
        t = self.trend
        c.print(f"  trend: [bold]{t['direction']}[/bold] ({t['change_per_year_pct']:+.1f}% per year, p={t['p_value']:.2g})")
        for s in self.seasonality:
            mark = "[green]yes[/]" if s["seasonal"] else "no"
            c.print(f"  {s['name']} seasonality (lag {s['lag']}): {mark}, strength {s['strength']:.2f}")
        if self.profile:
            peak = max(self.profile, key=self.profile.get)
            low = min(self.profile, key=self.profile.get)
            c.print(f"  profile: highest {peak}, lowest {low}")
        for a in self.anomalies:
            c.print(f"  [red]anomaly[/] {a['period']}: {a['value']:.4g} (robust z {a['z']:+.1f})")

    def __repr__(self) -> str:
        return (f"TimeSeriesReport({self.metric} per {self.freq}, periods={self.n_periods}, "
                f"trend={self.trend['direction']!r}, season={self.season_length})")


def timeseries(source: Any, time: str, value: Optional[str] = None, *,
               agg: str = "sum", freq: Optional[str] = None) -> TimeSeriesReport:
    """
    Profile `value` aggregated per period (or the row count when value is None).

    agg  : "sum" (events: sales, orders) or "mean" (measurements: temperature)
    freq : "hour" | "day" | "week" (default: chosen from the time span)
    """
    if agg not in ("sum", "mean"):
        raise ValueError("agg must be 'sum' or 'mean'")
    if freq is not None and freq not in FREQS:
        raise ValueError(f"freq must be one of {FREQS}")
    series, freq = aggregate(source, time, value, agg=agg, freq=freq)
    return _analyze(series, time, value, agg, freq)


# ── aggregation over the full data ───────────────────────────────────────────

def aggregate(source: Any, time: str, value: Optional[str] = None, *,
              agg: str = "sum", freq: Optional[str] = None):
    """(DataFrame[period, rows, value], freq) with one row per period, gaps filled."""
    if isinstance(source, (str, Path)) and is_available("duckdb"):
        raw, freq = _aggregate_duckdb(Path(source), time, value, agg, freq)
    else:
        raw, freq = _aggregate_pandas(_load(source, time, value), time, value, agg, freq)

    full = pd.date_range(raw["period"].min(), raw["period"].max(), freq=_offset(freq))
    out = raw.set_index("period").reindex(full)
    out.index.name = "period"
    out["rows"] = out["rows"].fillna(0).astype(int)
    if value is not None and agg == "sum":
        out["value"] = out["value"].fillna(0.0)
    return out.reset_index(), freq


def _aggregate_duckdb(path: Path, time: str, value, agg: str, freq):
    import duckdb

    con = duckdb.connect()
    ext = path.suffix.lower()
    if ext in PARQUET_EXTS:
        scan, params = "read_parquet(?)", [str(path)]
    elif ext in CSV_EXTS:
        scan, params = f"read_csv(?, delim = ?, header = true, {DUCKDB_CSV_TYPES})", [str(path), csv_sep(path)]
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    ts = f"TRY_CAST({_ident(time)} AS TIMESTAMP)"
    if freq is None:
        lo, hi = con.execute(f"SELECT min({ts}), max({ts}) FROM {scan}", params).fetchone()
        freq = _auto_freq(pd.Timestamp(lo), pd.Timestamp(hi))
    val = f", {agg}(TRY_CAST({_ident(value)} AS DOUBLE)) AS value" if value else ""
    df = con.execute(
        f"SELECT date_trunc('{freq}', {ts}) AS period, count(*) AS rows{val} "
        f"FROM {scan} WHERE {ts} IS NOT NULL GROUP BY 1 ORDER BY 1", params).df()
    df["period"] = pd.to_datetime(df["period"])
    return df, freq


def _aggregate_pandas(df: pd.DataFrame, time: str, value, agg: str, freq):
    ts = pd.to_datetime(df[time], errors="coerce", format="mixed")
    keep = ts.notna()
    ts = ts[keep]
    if freq is None:
        freq = _auto_freq(ts.min(), ts.max())
    period = ts.dt.to_period("W").dt.start_time if freq == "week" else ts.dt.floor(_offset(freq))
    frame = pd.DataFrame({"period": period.to_numpy()})
    if value is not None:
        frame["value"] = pd.to_numeric(df.loc[keep, value], errors="coerce").to_numpy()
    g = frame.groupby("period")
    out = g.size().rename("rows").to_frame()
    if value is not None:
        out["value"] = g["value"].sum(min_count=1) if agg == "sum" else g["value"].mean()
    return out.reset_index(), freq


def _load(source: Any, time: str, value) -> pd.DataFrame:
    cols = [time] + ([value] if value else [])
    if isinstance(source, pd.DataFrame):
        return source[cols]
    if _is_polars_frame(source):
        return source.select(cols).to_pandas()
    if _is_sql(source):
        quoted = ", ".join(_ident(c) for c in cols)
        return source.read(f"SELECT {quoted} FROM {source.from_sql()}")
    path = Path(source)
    if path.suffix.lower() in PARQUET_EXTS:
        return pd.read_parquet(path, columns=cols)
    return pd.read_csv(path, sep=csv_sep(path), usecols=cols)


def _auto_freq(lo: pd.Timestamp, hi: pd.Timestamp) -> str:
    span = hi - lo
    if span <= pd.Timedelta(days=60):
        return "hour"
    if span <= pd.Timedelta(days=5 * 365):
        return "day"
    return "week"


def _offset(freq: str) -> str:
    return {"hour": "h", "day": "D", "week": "W-MON"}[freq]


def _ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


# ── analysis of the aggregated series ────────────────────────────────────────

def _analyze(series: pd.DataFrame, time: str, value, agg: str, freq: str) -> TimeSeriesReport:
    y = (series["value"] if value else series["rows"]).astype(float)
    y_filled = y.interpolate(limit_direction="both") if y.isna().any() else y
    empty = (series["rows"] == 0).to_numpy()

    return TimeSeriesReport(
        time=time, value=value, agg=agg, freq=freq,
        start=series["period"].iloc[0], end=series["period"].iloc[-1],
        n_periods=len(series), missing_periods=int(empty.sum()), longest_gap=_longest_run(empty),
        rows_per_period=round(float(series["rows"].mean()), 2),
        trend=_trend(y_filled, freq),
        seasonality=_seasonality(y_filled, freq),
        profile=_profile(series["period"], y, freq),
        anomalies=_anomalies(series["period"], y),
        series=series,
    )


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def _trend(y: pd.Series, freq: str) -> Dict[str, Any]:
    from scipy.stats import linregress

    if len(y) < 3 or y.std() == 0:
        return {"direction": "flat", "slope": 0.0, "p_value": 1.0, "change_per_year_pct": 0.0}
    fit = linregress(np.arange(len(y)), y.to_numpy())
    mean = abs(float(y.mean())) or 1.0
    total_change = abs(fit.slope) * (len(y) - 1) / mean
    direction = "flat"
    if fit.pvalue < TREND_MAX_P and total_change >= TREND_MIN_CHANGE:
        direction = "rising" if fit.slope > 0 else "falling"
    return {
        "direction": direction,
        "slope": float(fit.slope),
        "p_value": float(fit.pvalue),
        "change_per_year_pct": round(float(fit.slope * PERIODS_PER_YEAR[freq] / mean * 100), 2),
    }


def _seasonality(y: pd.Series, freq: str) -> List[Dict[str, Any]]:
    x = y.to_numpy()
    if len(x) >= 3:
        t = np.arange(len(x))
        x = x - np.polyval(np.polyfit(t, x, 1), t)      # remove the trend first
    out = []
    for lag, name in SEASONAL_LAGS[freq]:
        if len(x) < 2 * lag + 1:
            continue
        a, b = x[:-lag], x[lag:]
        strength = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0
        out.append({"name": name, "lag": lag, "strength": round(strength, 3),
                    "seasonal": strength >= SEASONAL_MIN_ACF})
    return out


def _profile(period: pd.Series, y: pd.Series, freq: str) -> Dict[str, float]:
    if freq == "day":
        keys = period.dt.day_name()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    elif freq == "hour":
        keys = period.dt.hour.map(lambda h: f"{h:02d}:00")
        order = [f"{h:02d}:00" for h in range(24)]
    else:
        return {}
    means = y.groupby(keys.to_numpy()).mean()
    return {k: round(float(means[k]), 4) for k in order if k in means.index and pd.notna(means[k])}


def _anomalies(period: pd.Series, y: pd.Series, limit: int = 5) -> List[Dict[str, Any]]:
    v = y.dropna()
    if len(v) < 10:
        return []
    med = float(v.median())
    mad = float((v - med).abs().median()) * 1.4826
    if mad == 0:
        return []
    z = (y - med) / mad
    idx = z.abs().sort_values(ascending=False).index
    out = []
    for i in idx[:limit]:
        if pd.isna(z[i]) or abs(z[i]) < ANOMALY_Z:
            break
        out.append({"period": str(period[i]), "value": float(y[i]), "z": round(float(z[i]), 2)})
    return out
