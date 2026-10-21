from __future__ import annotations

from typing import Dict, List, Optional

from ..steps import (
    ClipOutliers, DateParts, DropColumns, DropDuplicates, DropNullRows, FillNull,
    FrequencyEncode, GroupRare, Log1p, MissingIndicator, NormalizeText, OneHot, Plan, Ratio,
    ToNumeric,
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
        def current(name: str) -> str:
            return exprs.get(name, f"pl.col({name!r})")

        def add(name: str, expr: str, reason: str) -> None:
            exprs[name] = f"({expr}).alias({name!r})"
            reasons.setdefault(name, []).append(reason)

        for step in steps:
            if isinstance(step, Ratio):
                num = f"{current(step.numerator)}.cast(pl.Float64)"
                den = f"{current(step.denominator)}.cast(pl.Float64)"
                add(step.name, f"pl.when({den} != 0).then({num} / {den})", step.reason)
                continue
            e = current(step.column)
            out = step.column
            if isinstance(step, DateParts):
                ts = f"{e}.str.to_datetime(strict=False)" if step.is_text else e
                for part, name in zip(step.parts, step.names):
                    fn = {"year": "year", "month": "month", "day": "day",
                          "dayofweek": "weekday", "hour": "hour"}[part]
                    add(name, f"{ts}.dt.{fn}().cast(pl.Float64)", step.reason)
                continue
            if isinstance(step, OneHot):
                for value, name in zip(step.values, step.names):
                    add(name, f"({e} == {value!r}).fill_null(False)", step.reason)
                continue
            if isinstance(step, FrequencyEncode):
                e = (f"{e}.replace_strict({dict(step.mapping)!r}, default={step.default!r}, "
                     f"return_dtype=pl.Float64).fill_null({step.default!r})")
            elif isinstance(step, ToNumeric):
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
