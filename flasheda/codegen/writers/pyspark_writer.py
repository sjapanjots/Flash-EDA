from __future__ import annotations

from typing import Optional

from ..steps import (
    ClipOutliers, DropColumns, DropDuplicates, DropNullRows, FillNull, GroupRare, Log1p,
    MissingIndicator, NormalizeText, Plan, ToNumeric,
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
            body.append(self._line(step))
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
        c = col(step.column)
        if isinstance(step, DropNullRows):
            return f"df = df.filter({c}.isNotNull())"
        if isinstance(step, MissingIndicator):
            return f"df = df.withColumn({step.name!r}, {c}.isNull())"
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
