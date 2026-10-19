from __future__ import annotations

from typing import List, Optional

from ..steps import (
    ClipOutliers, DropColumns, DropDuplicates, DropNullRows, FillNull, GroupRare, Log1p,
    MissingIndicator, NormalizeText, Plan, Step, ToNumeric,
)
from ...engine.source import DUCKDB_CSV_TYPES
from .base import Writer, source_format


def ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def literal(value) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        return f"CAST({value!r} AS DOUBLE)"
    return repr(value)


class DuckDBWriter(Writer):
    """Out-of-core SQL: DuckDB streams the file and spills to disk if needed."""

    engine = "duckdb"
    install_hint = "duckdb"

    def imports(self) -> str:
        return "import duckdb"

    def build_sql(self, plan: Plan) -> str:
        ctes: List[str] = []
        prev = "{table}"

        for kind, item in plan.segments():
            if kind == "row":
                select, where = self._row(item)
            else:
                select, where = self._columns(item), ""
            name = f"step_{len(ctes) + 1}"
            ctes.append(f"{name} AS (\n    {select} FROM {prev}{where}\n)")
            prev = name

        if not ctes:
            return "SELECT * FROM {table}"
        return "WITH " + ",\n".join(ctes) + f"\nSELECT * FROM {prev}"

    @staticmethod
    def _row(step: Step):
        if isinstance(step, DropColumns):
            return f"SELECT * EXCLUDE ({', '.join(ident(c) for c in step.columns)})", ""
        if isinstance(step, DropDuplicates):
            return "SELECT DISTINCT *", ""
        if isinstance(step, DropNullRows):
            return "SELECT *", f" WHERE {ident(step.column)} IS NOT NULL"
        raise TypeError(f"Unsupported step {step!r}")

    @staticmethod
    def _columns(steps: List[Step]) -> str:
        replaced = {}
        added = {}
        for step in steps:
            col = replaced.get(step.column, ident(step.column))
            if isinstance(step, ToNumeric):
                col = f"TRY_CAST(trim(CAST({col} AS VARCHAR), ' ') AS DOUBLE)"
            elif isinstance(step, NormalizeText):
                col = f"lower(trim({col}, ' '))"
            elif isinstance(step, GroupRare):
                keep = ", ".join(literal(v) for v in step.keep)
                col = (f"CASE WHEN {col} IS NULL OR {col} IN ({keep}) "
                       f"THEN {col} ELSE {literal(step.other)} END")
            elif isinstance(step, MissingIndicator):
                added[step.name] = f"({col} IS NULL)"
                continue
            elif isinstance(step, FillNull):
                col = f"coalesce({col}, {literal(step.value)})"
            elif isinstance(step, ClipOutliers):
                lo, hi = literal(float(step.lower)), literal(float(step.upper))
                col = f"CASE WHEN {col} < {lo} THEN {lo} WHEN {col} > {hi} THEN {hi} ELSE {col} END"
            elif isinstance(step, Log1p):
                col = f"sign({col}) * ln(1 + abs({col}))"
            replaced[step.column] = col

        sep = ",\n        "
        select = "SELECT *"
        if replaced:
            items = sep.join(f"{e} AS {ident(c)}" for c, e in replaced.items())
            select += f" REPLACE (\n        {items}\n    )"
        if added:
            select += ",\n    " + ",\n    ".join(f"{e} AS {ident(c)}" for c, e in added.items())
        return select

    def transform(self, plan: Plan) -> str:
        notes = [f"--   {d}" for d in plan.describe()]
        query = "\n".join((["-- Steps:"] + notes if notes else []) + [self.build_sql(plan)])
        query = query.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        return (
            f'QUERY = """\n{query}\n"""\n\n\n'
            "def build_query(table: str) -> str:\n"
            '    return QUERY.replace("{table}", table)\n\n\n'
            'def transform(con: duckdb.DuckDBPyConnection, table: str = "source"):\n'
            "    return con.sql(build_query(table))"
        )

    def main(self, source: Optional[str]) -> str:
        fmt = source_format(source)
        if fmt == "parquet":
            scan = "read_parquet('{src}')"
        else:
            sep = "\\t" if fmt == "tsv" else ","
            scan = f"read_csv('{{src}}', delim = '{sep}', header = true, {DUCKDB_CSV_TYPES})"
        return (
            "def main() -> None:\n"
            "    con = duckdb.connect()\n"
            "    src, out = (p.replace(\"'\", \"''\") for p in (SOURCE, OUTPUT))\n"
            f'    table = f"{scan}"\n'
            "    con.execute(f\"COPY ({build_query(table)}) TO '{out}' (FORMAT parquet)\")\n"
            '    print(f"Wrote -> {OUTPUT}")'
        )
