"""
flasheda.contract
~~~~~~~~~~~~~~~~~
Learn a data contract from a sample, then check new or full data against it:

    report = flasheda.analyze("orders_jan.csv")
    contract = report.contract()
    contract.save("orders.contract.json")

    result = flasheda.check("orders_feb.csv", "orders.contract.json")
    result.show()            # failures, if any
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from .check import CheckResult, Outcome
from .rules import ColumnRule, Contract, infer_contract

__all__ = ["CheckResult", "ColumnRule", "Contract", "Outcome", "check", "infer_contract"]


def check(source: Any, contract: Union[Contract, str, Path]) -> CheckResult:
    """Check `source` against a Contract or a saved contract JSON file."""
    if not isinstance(contract, Contract):
        contract = Contract.load(contract)
    return contract.check(source)
