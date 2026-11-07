"""
Sampling SQL for each database dialect. Only SQLite runs here; for the
others the generated SQL is checked as text (and parsed with sqlglot when
it's installed), since CI has no Postgres, MySQL, SQL Server or Oracle.
"""

import pandas as pd
import pytest

import flasheda as fe

sa = pytest.importorskip("sqlalchemy")

URLS = {
    "postgresql": "postgresql://u:secret@h/db",
    "mysql": "mysql://u:secret@h/db",
    "sqlite": "sqlite:///x.db",
    "mssql": "mssql+pyodbc://u:secret@h/db?driver=x",
    "oracle": "oracle://u:secret@h/db",
}

EXPECTED = {
    "postgresql": ['FROM "Sales"."Order Items" TABLESAMPLE BERNOULLI', "REPEATABLE (7) LIMIT 500"],
    "mysql": ["FROM `Sales`.`Order Items` ORDER BY rand() LIMIT 500"],
    "sqlite": ['FROM "Sales"."Order Items" ORDER BY random() LIMIT 500'],
    "mssql": ["SELECT TOP 500 * FROM [Sales].[Order Items] ORDER BY NEWID()"],
    "oracle": ['FROM "Sales"."Order Items" ORDER BY DBMS_RANDOM.VALUE FETCH FIRST 500 ROWS ONLY'],
}
SQLGLOT = {"postgresql": "postgres", "mysql": "mysql", "sqlite": "sqlite", "mssql": "tsql",
           "oracle": "oracle"}


@pytest.mark.parametrize("dialect", list(URLS))
def test_table_sample_sql(dialect):
    src = fe.sql(URLS[dialect], table="Sales.Order Items")
    assert src.dialect == dialect
    sql = src.sample_sql(500, random_state=7, total=100_000)
    for part in EXPECTED[dialect]:
        assert part in sql, sql
    assert "secret" not in src.display
    _parses(sql, dialect)


@pytest.mark.parametrize("dialect", list(URLS))
def test_query_sample_sql(dialect):
    src = fe.sql(URLS[dialect], query="SELECT a, b FROM t WHERE a > 1;")
    sql = src.sample_sql(100, total=1_000)
    assert "(SELECT a, b FROM t WHERE a > 1)" in sql and ";" not in sql
    assert (" AS flasheda_source" in sql) == (dialect != "oracle")   # Oracle: no AS on subqueries
    _parses(sql, dialect)


def test_postgres_without_count_falls_back_to_random_order():
    sql = fe.sql(URLS["postgresql"], table="t").sample_sql(10)
    assert sql.endswith("ORDER BY random() LIMIT 10")


def test_sqlite_runs_every_query(tmp_path):
    url = f"sqlite:///{tmp_path / 'd.db'}"
    pd.DataFrame({"a": range(1_000), "b": ["x"] * 1_000}).to_sql(
        "Order Items", sa.create_engine(url), index=False)
    src = fe.sql(url, table="Order Items")
    assert src.columns() == ["a", "b"]                   # WHERE 1 = 0: works in every dialect
    assert src.count() == 1_000
    assert len(src.sample(100)) == 100
    query = fe.sql(url, query='SELECT a FROM "Order Items" WHERE a < 300')
    assert query.columns() == ["a"] and query.count() == 300 and len(query.sample(50)) == 50


def _parses(sql, dialect):
    try:
        import sqlglot
    except ImportError:          # optional: installed with flasheda[dev]
        return
    sqlglot.parse_one(sql, read=SQLGLOT[dialect])
