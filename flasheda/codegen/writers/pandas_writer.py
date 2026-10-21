from __future__ import annotations

from typing import Optional

from ..steps import (
    ClipOutliers, DateParts, DropColumns, DropDuplicates, DropNullRows, FillNull,
    FrequencyEncode, GroupRare, Log1p, MissingIndicator, NormalizeText, OneHot, Plan, Ratio,
    ToNumeric,
)
from .base import Writer, source_format


class PandasWriter(Writer):
    engine = "pandas"
    install_hint = "pandas, pyarrow"

    def imports(self) -> str:
        return "import numpy as np\nimport pandas as pd"

    def transform(self, plan: Plan) -> str:
        body = []
        for step in plan:
            body.append(self.comment(step))
            body.extend(self._line(step).split("\n"))
        body.append("return df")
        return (
            "def transform(df: pd.DataFrame) -> pd.DataFrame:\n"
            + "    df = df.copy()\n"
            + self.indent([line for line in body if line])
        )

    @staticmethod
    def _line(step) -> str:
        if isinstance(step, DropColumns):
            return f"df = df.drop(columns={list(step.columns)!r})"
        if isinstance(step, DropDuplicates):
            return "df = df.drop_duplicates().reset_index(drop=True)"
        if isinstance(step, Ratio):
            den = f'df[{step.denominator!r}].astype("float64")'
            return (f'df[{step.name!r}] = df[{step.numerator!r}].astype("float64") / '
                    f"{den}.where({den} != 0)")
        c = f"df[{step.column!r}]"
        if isinstance(step, DropNullRows):
            return f"df = df[{c}.notna()].reset_index(drop=True)"
        if isinstance(step, ToNumeric):
            return (f'{c} = pd.to_numeric({c}.astype("string").str.strip(" "), '
                    'errors="coerce").astype("float64")')
        if isinstance(step, NormalizeText):
            return f'{c} = {c}.str.strip(" ").str.lower()'
        if isinstance(step, GroupRare):
            return f"{c} = {c}.where({c}.isin({list(step.keep)!r}) | {c}.isna(), {step.other!r})"
        if isinstance(step, MissingIndicator):
            return f"df[{step.name!r}] = {c}.isna()"
        if isinstance(step, FillNull):
            return f"{c} = {c}.fillna({step.value!r})"
        if isinstance(step, ClipOutliers):
            return f"{c} = {c}.clip({step.lower!r}, {step.upper!r})"
        if isinstance(step, Log1p):
            return f"{c} = np.sign({c}) * np.log1p(np.abs({c}))"
        if isinstance(step, DateParts):
            src = f'pd.to_datetime({c}, errors="coerce", format="ISO8601")' if step.is_text else c
            lines = [f"ts = {src}"]
            for part, name in zip(step.parts, step.names):
                value = "ts.dt.dayofweek + 1" if part == "dayofweek" else f"ts.dt.{part}"
                lines.append(f'df[{name!r}] = ({value}).astype("float64")')
            return "\n".join(lines)
        if isinstance(step, OneHot):
            return "\n".join(
                f"df[{name!r}] = {c}.eq({value!r}).fillna(False).astype(bool)"
                for value, name in zip(step.values, step.names))
        if isinstance(step, FrequencyEncode):
            return f'{c} = {c}.map({dict(step.mapping)!r}).astype("float64").fillna({step.default!r})'
        raise TypeError(f"Unsupported step {step!r}")

    def main(self, source: Optional[str]) -> str:
        fmt = source_format(source)
        if fmt == "parquet":
            load = "pd.read_parquet(SOURCE)"
        else:
            sep = "\\t" if fmt == "tsv" else ","
            load = f'pd.read_csv(SOURCE, sep="{sep}", low_memory=False)'
        return (
            "def main() -> None:\n"
            f"    df = transform({load})\n"
            "    df.to_parquet(OUTPUT, index=False)\n"
            '    print(f"Wrote {len(df):,} rows -> {OUTPUT}")'
        )
