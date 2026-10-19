from __future__ import annotations

from typing import Optional

from ..steps import ClipOutliers, DropColumns, DropDuplicates, FillNull, Log1p, Plan
from .base import Writer, source_format


class PolarsWriter(Writer):
    """Lazy polars: the query is optimised as a whole and streamed to disk."""

    engine = "polars"
    install_hint = "polars"

    def imports(self) -> str:
        return "import polars as pl"

    def transform(self, plan: Plan) -> str:
        body = []
        for step in plan:
            if isinstance(step, DropColumns):
                body += [self.comment(step), f"lf = lf.drop({list(step.columns)!r})"]
            elif isinstance(step, DropDuplicates):
                body += [self.comment(step), "lf = lf.unique(maintain_order=True)"]

        # Column steps run after row-level steps, all in a single with_columns()
        # so polars can evaluate every column in parallel.
        exprs = {}
        reasons = {}
        for step in plan.column_steps():
            expr = exprs.get(step.column, f"pl.col({step.column!r})")
            if isinstance(step, FillNull):
                if isinstance(step.value, float):
                    expr += ".cast(pl.Float64)"
                expr += f".fill_null({step.value!r})"
            elif isinstance(step, ClipOutliers):
                expr += f".cast(pl.Float64).clip({step.lower!r}, {step.upper!r})"
            elif isinstance(step, Log1p):
                expr = f"({expr}.sign() * {expr}.abs().log1p())"
            exprs[step.column] = expr
            reasons.setdefault(step.column, []).append(step.reason)

        if exprs:
            body.append("lf = lf.with_columns(")
            for col, expr in exprs.items():
                why = "; ".join(r for r in reasons[col] if r)
                body.append(f"    {expr},{'  # ' + why if why else ''}")
            body.append(")")

        body.append("return lf")
        return (
            "def transform(lf: pl.LazyFrame) -> pl.LazyFrame:\n"
            + self.indent([line for line in body if line])
        )

    def main(self, source: Optional[str]) -> str:
        fmt = source_format(source)
        if fmt == "parquet":
            scan = "pl.scan_parquet(SOURCE)"
        else:
            sep = "\\t" if fmt == "tsv" else ","
            scan = f'pl.scan_csv(SOURCE, separator="{sep}", infer_schema_length=10_000'
            # polars doesn't unescape "" inside quoted header names — pass them explicitly
            if self.columns and any('"' in str(c) for c in self.columns):
                scan += f", new_columns={[str(c) for c in self.columns]!r}"
            scan += ")"
        return (
            "def main() -> None:\n"
            "    # Lazy scan + streaming sink: the full file is never held in memory.\n"
            f"    transform({scan}).sink_parquet(OUTPUT)\n"
            '    print(f"Wrote -> {OUTPUT}")'
        )
