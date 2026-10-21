from __future__ import annotations

from typing import Optional

from ..steps import (
    ClipOutliers, DateParts, DropColumns, DropDuplicates, DropNullRows, FillNull,
    FrequencyEncode, GroupRare, Log1p, MissingIndicator, NormalizeText, OneHot, Plan, Ratio,
    TextStats, ToNumeric,
)
from .base import Writer, source_format


def col(name: str) -> str:
    return "F.col(" + repr("`" + str(name).replace("`", "``") + "`") + ")"


class PySparkWriter(Writer):
    """Distributed execution for cluster-scale data."""

    engine = "pyspark"
    install_hint = "pyspark"

    def imports(self) -> str:
        return "from pyspark.sql import DataFrame, SparkSession\nfrom pyspark.sql import functions as F"

    def transform(self, plan: Plan) -> str:
        body = []
        for step in plan:
            body.append(self.comment(step))
            body.extend(self._line(step).split("\n"))
        body.append("return df")
        return "def transform(df: DataFrame) -> DataFrame:\n" + self.indent(
            [line for line in body if line]
        )

    @staticmethod
    def _line(step) -> str:
        if isinstance(step, DropColumns):
            return f"df = df.drop(*{list(step.columns)!r})"
        if isinstance(step, DropDuplicates):
            return "df = df.dropDuplicates()"
        if isinstance(step, Ratio):
            den = f'{col(step.denominator)}.cast("double")'
            return (f"df = df.withColumn({step.name!r}, "
                    f'{col(step.numerator)}.cast("double") / F.when({den} != 0, {den}))')
        c = col(step.column)
        if isinstance(step, DropNullRows):
            return f"df = df.filter({c}.isNotNull())"
        if isinstance(step, MissingIndicator):
            return f"df = df.withColumn({step.name!r}, {c}.isNull())"
        if isinstance(step, DateParts):
            ts = f"F.to_timestamp({c})" if step.is_text else c
            spark = {"year": "F.year", "month": "F.month", "day": "F.dayofmonth", "hour": "F.hour"}
            lines = []
            for part, name in zip(step.parts, step.names):
                if part == "dayofweek":   # Spark: Sunday = 1; ISO: Monday = 1
                    value = f"((F.dayofweek({ts}) + 5) % 7) + 1"
                else:
                    value = f"{spark[part]}({ts})"
                lines.append(f'df = df.withColumn({name!r}, ({value}).cast("double"))')
            return "\n".join(lines)
        if isinstance(step, TextStats):
            return (f'df = df.withColumn({step.length_name!r}, F.length({c}).cast("double"))\n'
                    f"df = df.withColumn({step.words_name!r}, "
                    f'F.size(F.regexp_extract_all({c}, F.lit(r"\\S+"), F.lit(0))).cast("double"))')
        if isinstance(step, OneHot):
            return "\n".join(
                f"df = df.withColumn({name!r}, F.coalesce({c} == {value!r}, F.lit(False)))"
                for value, name in zip(step.values, step.names))
        if isinstance(step, ToNumeric):
            e = f'F.trim({c}.cast("string")).cast("double")'
        elif isinstance(step, NormalizeText):
            e = f"F.lower(F.trim({c}))"
        elif isinstance(step, GroupRare):
            e = (f"F.when({c}.isNull() | {c}.isin({list(step.keep)!r}), {c})"
                 f".otherwise(F.lit({step.other!r}))")
        elif isinstance(step, FillNull):
            e = f"F.coalesce({c}, F.lit({step.value!r}))"
        elif isinstance(step, ClipOutliers):
            e = (f"F.when({c} < {step.lower!r}, F.lit({step.lower!r}))"
                 f".when({c} > {step.upper!r}, F.lit({step.upper!r})).otherwise({c})")
        elif isinstance(step, Log1p):
            e = f"F.signum({c}) * F.log1p(F.abs({c}))"
        elif isinstance(step, FrequencyEncode):
            pairs = ", ".join(f"F.lit({v!r}), F.lit({float(f)!r})" for v, f in step.mapping)
            e = f"F.coalesce(F.create_map({pairs})[{c}], F.lit({step.default!r}))"
        else:
            raise TypeError(f"Unsupported step {step!r}")
        return f"df = df.withColumn({step.column!r}, {e})"

    def main(self, source: Optional[str]) -> str:
        fmt = source_format(source)
        if fmt == "parquet":
            load = "spark.read.parquet(SOURCE)"
        else:
            sep = "\\t" if fmt == "tsv" else ","
            load = f'spark.read.csv(SOURCE, header=True, inferSchema=True, sep="{sep}")'
        return (
            "def main() -> None:\n"
            '    spark = SparkSession.builder.appName("flasheda-clean").getOrCreate()\n'
            f"    transform({load}).write.mode(\"overwrite\").parquet(OUTPUT)\n"
            '    print(f"Wrote -> {OUTPUT}")'
        )
