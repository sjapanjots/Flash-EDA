"""
flasheda.codegen.steps
~~~~~~~~~~~~~~~~~~~~~~
Engine-neutral transformation steps. A `Plan` is an ordered list of
steps; writers turn it into pandas / polars / DuckDB SQL / PySpark code.

All parameters (fill values, clip bounds, kept categories) are literals
learned from the sample, so every engine produces exactly the same result.

`Plan.apply()` is a plain-pandas reference implementation used to check
that the generated code for every engine is correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import pandas as pd


# ── Row-level steps ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DropColumns:
    columns: tuple
    reason: str = ""


@dataclass(frozen=True)
class DropDuplicates:
    reason: str = ""


@dataclass(frozen=True)
class DropNullRows:
    """Drop rows where `column` is null (used for the target)."""

    column: str
    reason: str = ""


# ── Column-level steps ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToNumeric:
    """Parse text as numbers (spaces trimmed); unparseable values become null."""

    column: str
    reason: str = ""


@dataclass(frozen=True)
class NormalizeText:
    """Trim spaces and lower-case, so "Delhi", " delhi" and "DELHI" match."""

    column: str
    reason: str = ""


@dataclass(frozen=True)
class GroupRare:
    """Replace every non-null value not in `keep` with `other`."""

    column: str
    keep: tuple
    other: str = "other"
    reason: str = ""


@dataclass(frozen=True)
class MissingIndicator:
    """Add a boolean column `name` that is True where `column` is null."""

    column: str
    name: str
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


ROW_STEPS = (DropColumns, DropDuplicates, DropNullRows)
COLUMN_STEPS = (ToNumeric, NormalizeText, GroupRare, MissingIndicator, FillNull, ClipOutliers, Log1p)

Step = Union[
    DropColumns, DropDuplicates, DropNullRows,
    ToNumeric, NormalizeText, GroupRare, MissingIndicator, FillNull, ClipOutliers, Log1p,
]


@dataclass
class Plan:
    steps: List[Step] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def column_steps(self) -> List[Step]:
        return [s for s in self.steps if isinstance(s, COLUMN_STEPS)]

    def segments(self):
        """
        Split the plan into ("row", step) and ("columns", [steps]) segments.
        Consecutive column steps form one segment so engines can evaluate
        them together; row steps keep their position in the order.
        """
        pending: List[Step] = []
        for step in self.steps:
            if isinstance(step, COLUMN_STEPS):
                pending.append(step)
                continue
            if pending:
                yield "columns", pending
                pending = []
            yield "row", step
        if pending:
            yield "columns", pending

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
            elif isinstance(step, DropNullRows):
                out = out[out[step.column].notna()].reset_index(drop=True)
            elif isinstance(step, ToNumeric):
                text = out[step.column].astype("string").str.strip(" ")
                out[step.column] = pd.to_numeric(text, errors="coerce").astype("float64")
            elif isinstance(step, NormalizeText):
                out[step.column] = out[step.column].str.strip(" ").str.lower()
            elif isinstance(step, GroupRare):
                s = out[step.column]
                out[step.column] = s.where(s.isin(list(step.keep)) | s.isna(), step.other)
            elif isinstance(step, MissingIndicator):
                out[step.name] = out[step.column].isna()
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
    if isinstance(step, DropNullRows):
        return f"drop rows where {step.column!r} is null"
    if isinstance(step, ToNumeric):
        return f"convert {step.column!r} from text to numbers"
    if isinstance(step, NormalizeText):
        return f"trim and lower-case {step.column!r}"
    if isinstance(step, GroupRare):
        return f"group rare values of {step.column!r} into {step.other!r} (keep {len(step.keep)})"
    if isinstance(step, MissingIndicator):
        return f"add {step.name!r} flag for missing {step.column!r}"
    if isinstance(step, FillNull):
        return f"fill nulls in {step.column!r} with {step.value!r}"
    if isinstance(step, ClipOutliers):
        return f"clip {step.column!r} to [{step.lower}, {step.upper}]"
    return f"log transform {step.column!r}"
