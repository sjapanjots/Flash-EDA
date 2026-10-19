"""
flasheda.codegen.steps
~~~~~~~~~~~~~~~~~~~~~~
Engine-neutral transformation steps. A `Plan` is an ordered list of
steps; writers turn it into pandas / polars / DuckDB SQL / PySpark code.

All parameters (fill values, clip bounds) are literals learned from the
sample, so every engine produces exactly the same result.

`Plan.apply()` is a plain-pandas reference implementation used to check
that the generated code for every engine is correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DropColumns:
    columns: tuple
    reason: str = ""


@dataclass(frozen=True)
class DropDuplicates:
    reason: str = ""


@dataclass(frozen=True)
class FillNull:
    column: str
    value: Union[int, float, str]
    reason: str = ""


@dataclass(frozen=True)
class ClipOutliers:
    column: str
    lower: float
    upper: float
    reason: str = ""


@dataclass(frozen=True)
class Log1p:
    """Sign-preserving log: sign(x) * log1p(|x|).

    Identical to log1p for x >= 0, and still defined if the full dataset
    holds negative values the sample never saw.
    """

    column: str
    reason: str = ""


Step = Union[DropColumns, DropDuplicates, FillNull, ClipOutliers, Log1p]


@dataclass
class Plan:
    steps: List[Step] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def column_steps(self) -> List[Step]:
        return [s for s in self.steps if isinstance(s, (FillNull, ClipOutliers, Log1p))]

    def describe(self) -> List[str]:
        return [f"{_label(s)} — {s.reason}" if s.reason else _label(s) for s in self.steps]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reference implementation in pandas."""
        out = df.copy()
        for step in self.steps:
            if isinstance(step, DropColumns):
                out = out.drop(columns=list(step.columns))
            elif isinstance(step, DropDuplicates):
                out = out.drop_duplicates().reset_index(drop=True)
            elif isinstance(step, FillNull):
                out[step.column] = out[step.column].fillna(step.value)
            elif isinstance(step, ClipOutliers):
                out[step.column] = out[step.column].clip(step.lower, step.upper)
            elif isinstance(step, Log1p):
                x = out[step.column]
                out[step.column] = np.sign(x) * np.log1p(np.abs(x))
        return out


def _label(step: Step) -> str:
    if isinstance(step, DropColumns):
        return f"drop columns {list(step.columns)}"
    if isinstance(step, DropDuplicates):
        return "drop duplicate rows"
    if isinstance(step, FillNull):
        return f"fill nulls in {step.column!r} with {step.value!r}"
    if isinstance(step, ClipOutliers):
        return f"clip {step.column!r} to [{step.lower}, {step.upper}]"
    return f"log transform {step.column!r}"
