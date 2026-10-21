"""
flasheda.codegen
~~~~~~~~~~~~~~~~
Turns EDA findings into runnable, engine-specific cleaning code.

    findings → Plan (engine-neutral steps) → writer → Python script
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd

from .planner import plan_from_report
from .steps import (
    ClipOutliers, DateParts, DropColumns, DropDuplicates, DropNullRows, FillNull,
    FrequencyEncode, GroupRare, Log1p, MissingIndicator, NormalizeText, OneHot, Plan, Ratio,
    ToNumeric,
)
from .validate import ValidationResult, compare_frames, run_generated, validate_code
from .writers import WRITERS, get_writer


@dataclass
class GeneratedCode:
    engine: str
    code: str
    plan: Plan
    sample: Optional[pd.DataFrame] = field(default=None, repr=False)
    report: Any = field(default=None, repr=False)   # set by flasheda.clean()

    def __str__(self) -> str:
        return self.code

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.write_text(self.code, encoding="utf-8")
        return p

    def validate(self) -> ValidationResult:
        if self.sample is None:
            raise ValueError("No sample attached — nothing to validate against.")
        return validate_code(self.code, self.engine, self.plan, self.sample)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the generated transform on an in-memory DataFrame (returns pandas)."""
        return run_generated(self.code, self.engine, df)


def generate(
    plan: Plan,
    engine: str = "pandas",
    *,
    source: Optional[str] = None,
    output: str = "cleaned.parquet",
    sample: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
) -> GeneratedCode:
    if columns is None and sample is not None:
        columns = [str(c) for c in sample.columns]
    code = get_writer(engine).render(plan, source, output, columns)
    return GeneratedCode(engine, code, plan, sample)


__all__ = [
    "ClipOutliers",
    "DateParts",
    "DropColumns",
    "DropDuplicates",
    "DropNullRows",
    "FillNull",
    "FrequencyEncode",
    "GeneratedCode",
    "GroupRare",
    "Log1p",
    "MissingIndicator",
    "NormalizeText",
    "OneHot",
    "Plan",
    "Ratio",
    "ToNumeric",
    "ValidationResult",
    "WRITERS",
    "compare_frames",
    "generate",
    "plan_from_report",
    "validate_code",
]
