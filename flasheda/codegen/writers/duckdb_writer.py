from __future__ import annotations

from typing import List, Optional

from ..steps import ClipOutliers, DropColumns, DropDuplicates, FillNull, Log1p, Plan, Step
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
        pending: List[Step] = []

        def add(select: str) -> None:
            nonlocal prev
            name = f"step_{len(ctes) + 1}"
            ctes.append(f"{name} AS (\n    {select} FROM {prev}\n)")
            prev = name

        def flush() -> None:
            if pending:
                add(self._replace(pending))
                pending.clear()

        for step in plan:
            if isinstance(step, DropColumns):
                flush()
                add(f"SELECT * EXCLUDE ({', '.join(ident(c) for c in step.columns)})")
            elif isinstance(step, DropDuplicates):
                flush()
                add("SELECT DISTINCT *")
            else:
                pending.append(step)
        flush()

        if not ctes:
            return "SELECT * FROM {table}"
        return "WITH " + ",\n".join(ctes) + f"\nSELECT * FROM {prev}"

    def _replace(self, steps: List[Step]) -> str:
        exprs = {}
        for step in steps:
            col = exprs.get(step.column, ident(step.column))
            if isinstance(step, FillNull):
                col = f"coalesce({col}, {literal(step.value)})"
            elif isinstance(step, ClipOutliers):
                lo, hi = literal(float(step.lower)), literal(float(step.upper))
                col = f"CASE WHEN {col} < {lo} THEN {lo} WHEN {col} > {hi} THEN {hi} ELSE {col} END"
            elif isinstance(step, Log1p):
                col = f"sign({col}) * ln(1 + abs({col}))"
            exprs[step.column] = col
        items = ",\n        ".join(f"{e} AS {ident(c)}" for c, e in exprs.items())
        return f"SELECT * REPLACE (\n        {items}\n    )"

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
            scan = f"read_csv('{{src}}', delim = '{sep}', header = true)"
        return (
            "def main() -> None:\n"
            "    con = duckdb.connect()\n"
            "    src, out = (p.replace(\"'\", \"''\") for p in (SOURCE, OUTPUT))\n"
            f'    table = f"{scan}"\n'
            "    con.execute(f\"COPY ({build_query(table)}) TO '{out}' (FORMAT parquet)\")\n"
            '    print(f"Wrote -> {OUTPUT}")'
        )
