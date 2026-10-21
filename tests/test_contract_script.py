import re
import subprocess
import sys

import pandas as pd
import pytest

import flasheda as fe
from test_contract import EXPECTED_FAILURE, _break, contract, months  # noqa: F401  (fixtures)


def run_script(code: str, tmp_path, data) -> subprocess.CompletedProcess:
    script = tmp_path / "check.py"
    script.write_text(code)
    return subprocess.run([sys.executable, "-I", str(script), str(data)],
                          capture_output=True, text=True, timeout=120)


def failed_checks(stdout: str) -> set:
    out = set()
    for line in stdout.splitlines():
        m = re.match(r"\[FAIL\] (?:(.+?): )?(.+?): observed", line)
        if m:
            col, rule = m.groups()
            out.add((col, rule) if col and not rule.startswith("required") else (None, rule))
    return out


@pytest.mark.parametrize("ext", ["csv", "parquet"])
def test_script_passes_similar_data(contract, months, tmp_path, ext):
    res = run_script(contract.script(), tmp_path, months / f"feb.{ext}")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "checks passed on 12,100 rows" in res.stdout
    assert "[FAIL]" not in res.stdout


def test_script_uses_default_source(contract, months, tmp_path):
    script = tmp_path / "check.py"
    script.write_text(contract.script(str(months / "feb.csv")))
    assert subprocess.run([sys.executable, "-I", str(script)], capture_output=True).returncode == 0


@pytest.mark.parametrize("how", list(EXPECTED_FAILURE))
def test_script_matches_flasheda_check(contract, months, tmp_path, how):
    broken = _break(pd.read_csv(months / "feb.csv", dtype={"price": str}), how)
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)

    res = run_script(contract.script(), tmp_path, path)
    assert res.returncode == 1
    expected = {(f.column, f.rule) for f in fe.check(path, contract).failures}
    assert failed_checks(res.stdout) == expected == EXPECTED_FAILURE[how]


# ── pandera export ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def schema(contract):
    pytest.importorskip("pandera.pandas")
    ns: dict = {}
    exec(contract.to_pandera(), ns)
    return ns["schema"]


def test_pandera_passes_similar_data(schema, months):
    schema.validate(pd.read_csv(months / "feb.csv", dtype={"price": str}), lazy=True)


@pytest.mark.parametrize("how", list(EXPECTED_FAILURE))
def test_pandera_catches_breaks(schema, months, how):
    import pandera.errors as pe

    broken = _break(pd.read_csv(months / "feb.csv", dtype={"price": str}), how)
    with pytest.raises(pe.SchemaErrors) as err:
        schema.validate(broken, lazy=True)
    cases = err.value.failure_cases
    if how == "missing column":       # schema-level: the name is in failure_case
        assert "source" in set(cases["failure_case"].astype(str))
    else:
        expected = {c for c, _ in EXPECTED_FAILURE[how]}
        assert expected <= set(cases["column"].dropna().astype(str))
