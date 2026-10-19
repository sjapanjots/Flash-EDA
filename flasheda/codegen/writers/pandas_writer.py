from __future__ import annotations

from typing import Optional

from ..steps import ClipOutliers, DropColumns, DropDuplicates, FillNull, Log1p, Plan
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
            if isinstance(step, DropColumns):
                body.append(f"df = df.drop(columns={list(step.columns)!r})")
            elif isinstance(step, DropDuplicates):
                body.append("df = df.drop_duplicates().reset_index(drop=True)")
            elif isinstance(step, FillNull):
                body.append(f"df[{step.column!r}] = df[{step.column!r}].fillna({step.value!r})")
            elif isinstance(step, ClipOutliers):
                body.append(
                    f"df[{step.column!r}] = df[{step.column!r}].clip({step.lower!r}, {step.upper!r})"
                )
            elif isinstance(step, Log1p):
                c = f"df[{step.column!r}]"
                body.append(f"{c} = np.sign({c}) * np.log1p(np.abs({c}))")
        body.append("return df")
        return (
            "def transform(df: pd.DataFrame) -> pd.DataFrame:\n"
            + "    df = df.copy()\n"
            + self.indent([line for line in body if line])
        )

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
