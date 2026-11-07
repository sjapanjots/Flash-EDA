"""
flasheda.codegen.registry
~~~~~~~~~~~~~~~~~~~~~~~~~
What each step does to your data, as data: how risky it is, whether it
can be undone from the output, and which engines can generate it.

    risk   low     adds columns or a reversible transform (log)
           medium  changes values in place (parse, fill, group, lower-case)
           high    removes data (columns, rows) or caps values

The planner uses this with each step's confidence (how sure the sample is
about the decision) to decide which steps need your approval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Type

from .steps import (
    ClipOutliers, DateParts, DropColumns, DropDuplicates, DropNullRows, FillNull,
    FrequencyEncode, GroupRare, Log1p, MissingIndicator, NormalizeText, OneHot, Ratio,
    TextStats, ToNumeric,
)

ENGINES = ("pandas", "polars", "duckdb", "pyspark")
RISKS = ("low", "medium", "high")
CONFIDENT = 0.95          # high-risk steps below this need approval (approve="confident")


@dataclass(frozen=True)
class StepInfo:
    risk: str
    reversible: bool
    effect: str
    engines: tuple = ENGINES


REGISTRY: Dict[Type, StepInfo] = {
    DropColumns:      StepInfo("high", False, "removes columns"),
    DropDuplicates:   StepInfo("high", False, "removes rows"),
    DropNullRows:     StepInfo("low", False, "removes rows without a target"),
    ClipOutliers:     StepInfo("high", False, "caps values"),
    ToNumeric:        StepInfo("medium", False, "unparseable text becomes null"),
    NormalizeText:    StepInfo("medium", False, "loses case and spacing"),
    GroupRare:        StepInfo("medium", False, "merges values"),
    FillNull:         StepInfo("medium", False, "replaces nulls"),
    FrequencyEncode:  StepInfo("medium", False, "replaces values with shares"),
    Log1p:            StepInfo("low", True, "transforms values (undo: sign(x)·expm1(|x|))"),
    MissingIndicator: StepInfo("low", True, "adds a column"),
    DateParts:        StepInfo("low", True, "adds columns"),
    Ratio:            StepInfo("low", True, "adds a column"),
    OneHot:           StepInfo("low", True, "adds columns"),
    TextStats:        StepInfo("low", True, "adds columns"),
}


def info(step) -> StepInfo:
    try:
        return REGISTRY[type(step)]
    except KeyError:
        raise TypeError(f"Unknown step type {type(step).__name__}: register it in "
                        "flasheda.codegen.registry.REGISTRY") from None


def register(step_type: Type, step_info: StepInfo) -> None:
    """Register a custom step type (its writers must handle it too)."""
    if step_info.risk not in RISKS:
        raise ValueError(f"risk must be one of {RISKS}")
    REGISTRY[step_type] = step_info


def margin_confidence(value: float, threshold: float, se: float) -> float:
    """
    Probability-like confidence that `value`, measured on a sample with
    standard error `se`, is on the same side of `threshold` in the full data:
    the normal CDF of the margin in standard errors.
    """
    if se <= 0:
        return 1.0
    z = abs(value - threshold) / se
    return round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)


def proportion_confidence(p: float, threshold: float, n: int) -> float:
    """margin_confidence for a share `p` measured on `n` rows."""
    return margin_confidence(p, threshold, math.sqrt(threshold * (1 - threshold) / max(n, 1)))
