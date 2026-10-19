from __future__ import annotations

from typing import Optional

from ..steps import ClipOutliers, DropColumns, DropDuplicates, FillNull, Log1p, Plan
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
            if isinstance(step, DropColumns):
                body.append(f"df = df.drop(*{list(step.columns)!r})")
            elif isinstance(step, DropDuplicates):
                body.append("df = df.dropDuplicates()")
            elif isinstance(step, FillNull):
                body.append(
                    f"df = df.withColumn({step.column!r}, "
                    f"F.coalesce({col(step.column)}, F.lit({step.value!r})))"
                )
            elif isinstance(step, ClipOutliers):
                c = col(step.column)
                body.append(
                    f"df = df.withColumn({step.column!r}, "
                    f"F.when({c} < {step.lower!r}, F.lit({step.lower!r}))"
                    f".when({c} > {step.upper!r}, F.lit({step.upper!r}))"
                    f".otherwise({c}))"
                )
            elif isinstance(step, Log1p):
                c = col(step.column)
                body.append(f"df = df.withColumn({step.column!r}, F.signum({c}) * F.log1p(F.abs({c})))")
        body.append("return df")
        return "def transform(df: DataFrame) -> DataFrame:\n" + self.indent(
            [line for line in body if line]
        )

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
