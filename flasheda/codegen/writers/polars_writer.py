from __future__ import annotations

from typing import Dict, List, Optional

from ..steps import (
    ClipOutliers, DropColumns, DropDuplicates, DropNullRows, FillNull, GroupRare, Log1p,
    MissingIndicator, NormalizeText, Plan, ToNumeric,
)
from .base import Writer, source_format


class PolarsWriter(Writer):
    """Lazy polars: the query is optimised as a whole and streamed to disk."""

    engine = "polars"
    install_hint = "polars"

    def imports(self) -> str:
        return "import polars as pl"

    def transform(self, plan: Plan) -> str:
        body: List[str] = []
        for kind, item in plan.segments():
            if kind == "row":
                body += [self.comment(item), self._row(item)]
            else:
                body += self._columns(item)
        body.append("return lf")
        return (
            "def transform(lf: pl.LazyFrame) -> pl.LazyFrame:\n"
            + self.indent([line for line in body if line])
        )

    @staticmethod
    def _row(step) -> str:
        if isinstance(step, DropColumns):
            return f"lf = lf.drop({list(step.columns)!r})"
        if isinstance(step, DropDuplicates):
            return "lf = lf.unique(maintain_order=True)"
        if isinstance(step, DropNullRows):
            return f"lf = lf.filter(pl.col({step.column!r}).is_not_null())"
        raise TypeError(f"Unsupported step {step!r}")

    def _columns(self, steps) -> List[str]:
        # One with_columns() per segment so polars evaluates the columns in
        # parallel. Every expression reads the segment's input, so steps on the
        # same column are composed into a single expression, in order.
        exprs: Dict[str, str] = {}
        reasons: Dict[str, List[str]] = {}
        for step in steps:
            e = exprs.get(step.column, f"pl.col({step.column!r})")
            out = step.column
            if isinstance(step, ToNumeric):
                e += '.cast(pl.String).str.strip_chars(" ").cast(pl.Float64, strict=False)'
            elif isinstance(step, NormalizeText):
                e += '.str.strip_chars(" ").str.to_lowercase()'
            elif isinstance(step, GroupRare):
                e = (f"pl.when({e}.is_in({list(step.keep)!r}) | {e}.is_null())"
                     f".then({e}).otherwise(pl.lit({step.other!r}))")
            elif isinstance(step, MissingIndicator):
                out = step.name
                exprs[out] = f"{e}.is_null().alias({step.name!r})"
                reasons.setdefault(out, []).append(step.reason)
                continue
            elif isinstance(step, FillNull):
                if isinstance(step.value, float):
                    e += ".cast(pl.Float64)"
                e += f".fill_null({step.value!r})"
            elif isinstance(step, ClipOutliers):
                e += f".cast(pl.Float64).clip({step.lower!r}, {step.upper!r})"
            elif isinstance(step, Log1p):
                e = f"({e}.sign() * {e}.abs().log1p())"
            exprs[out] = e
            reasons.setdefault(out, []).append(step.reason)

        lines = ["lf = lf.with_columns("]
        for name, e in exprs.items():
            why = "; ".join(r for r in reasons[name] if r)
            lines.append(f"    {e},{'  # ' + why if why else ''}")
        lines.append(")")
        return lines

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
