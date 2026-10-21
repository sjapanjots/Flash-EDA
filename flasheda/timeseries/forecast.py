"""
flasheda.timeseries.forecast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Baseline forecasts for an aggregated series, compared by rolling-origin
backtests (train on the past, score the next `horizon` periods, repeat):

  naive            last value
  seasonal_naive   value one season ago
  moving_average   mean of the last season (or 7 periods)
  drift            straight line through the first and last values
  trend_seasonal   linear trend + average seasonal pattern of the residuals

Simple on purpose: these are the baselines any model has to beat. The
winner is rendered as a standalone script that aggregates the full data
and writes the forecast as CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from . import TimeSeriesReport

DEFAULT_HORIZON = {"hour": 24, "day": 14, "week": 8}
FOLDS = 3


# ── methods (the same source is embedded in generated scripts) ──────────────

METHODS_SRC = {
    "naive": '''def forecast(y, h, m):
    return np.repeat(y[-1], h)''',
    "seasonal_naive": '''def forecast(y, h, m):
    last = y[-m:]
    return np.array([last[i % m] for i in range(h)])''',
    "moving_average": '''def forecast(y, h, m):
    return np.repeat(y[-(m or 7):].mean(), h)''',
    "drift": '''def forecast(y, h, m):
    slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
    return y[-1] + slope * np.arange(1, h + 1)''',
    "trend_seasonal": '''def forecast(y, h, m):
    t = np.arange(len(y))
    a, b = np.polyfit(t, y, 1)
    resid = y - (a * t + b)
    season = np.array([resid[(t % m) == k].mean() for k in range(m)])
    future = np.arange(len(y), len(y) + h)
    return a * future + b + season[future % m]''',
}


def _compile(name: str) -> Callable:
    ns: Dict[str, Any] = {"np": np}
    exec(METHODS_SRC[name], ns)
    return ns["forecast"]


METHODS = {name: _compile(name) for name in METHODS_SRC}
NEEDS_SEASON = {"seasonal_naive", "trend_seasonal"}


@dataclass
class ForecastResult:
    method: str
    horizon: int
    season: Optional[int]
    leaderboard: pd.DataFrame
    forecast: pd.DataFrame
    report: Any = field(repr=False)

    def script(self, source: Optional[str] = None, output: str = "forecast.csv") -> str:
        from .script import render_forecast_script
        return render_forecast_script(self, source, output)

    def show(self) -> None:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Backtest: {FOLDS} folds × {self.horizon} periods "
                            f"({self.report.metric} per {self.report.freq})", box=box.SIMPLE)
        for col in ("method", "MAE", "sMAPE %"):
            table.add_column(col)
        for _, row in self.leaderboard.iterrows():
            style = "bold green" if row["method"] == self.method else None
            table.add_row(row["method"], f"{row['mae']:.4g}", f"{row['smape']:.1f}", style=style)
        Console().print(table)

    def __repr__(self) -> str:
        best = self.leaderboard.iloc[0]
        return (f"ForecastResult(method={self.method!r}, horizon={self.horizon}, "
                f"MAE={best['mae']:.4g}, sMAPE={best['smape']:.1f}%)")


def forecast(report: "TimeSeriesReport", horizon: Optional[int] = None,
             methods: Optional[List[str]] = None) -> ForecastResult:
    y = series_values(report)
    h = int(horizon or DEFAULT_HORIZON[report.freq])
    m = report.season_length
    names = [n for n in (methods or list(METHODS)) if n in METHODS and (m or n not in NEEDS_SEASON)]
    if not names:
        raise ValueError(f"No usable methods in {methods!r}; seasonal methods need a detected season.")
    if len(y) < h * (FOLDS + 1):
        raise ValueError(f"Series too short ({len(y)} periods) for {FOLDS} backtests of {h} periods.")

    rows = []
    for name in names:
        errors, smapes = [], []
        for k in range(FOLDS, 0, -1):
            cut = len(y) - k * h
            pred = METHODS[name](y[:cut], h, m)
            actual = y[cut:cut + h]
            errors.append(np.abs(pred - actual).mean())
            denom = (np.abs(pred) + np.abs(actual)) / 2
            smapes.append(np.mean(np.where(denom > 0, np.abs(pred - actual) / np.where(denom > 0, denom, 1), 0)) * 100)
        rows.append({"method": name, "mae": float(np.mean(errors)), "smape": float(np.mean(smapes))})
    board = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)

    best = board.iloc[0]["method"]
    future = pd.date_range(report.end, periods=h + 1, freq=_offset(report.freq))[1:]
    values = METHODS[best](y, h, m)
    return ForecastResult(best, h, m, board,
                          pd.DataFrame({"period": future, "forecast": values}), report)


def series_values(report: "TimeSeriesReport") -> np.ndarray:
    """The aggregated series as floats, with empty periods filled by interpolation."""
    s = report.series["value"] if report.value else report.series["rows"]
    s = s.astype(float)
    return s.interpolate(limit_direction="both").to_numpy() if s.isna().any() else s.to_numpy()


def _offset(freq: str) -> str:
    return {"hour": "h", "day": "D", "week": "W-MON"}[freq]
