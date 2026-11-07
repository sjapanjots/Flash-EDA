"""
flasheda.config
~~~~~~~~~~~~~~~
All cleaning choices in one object you can keep, share and version:

    cfg = fe.CleanConfig(protect=["order_id"], clip_outliers=True, decimal=",")
    cfg.save("cleaning.json")

    report.plan(config=cfg)
    fe.clean("orders.csv", config=fe.CleanConfig.load("cleaning.json"))
    fe.train("orders.csv", target="churn", config=cfg)

Keyword options passed next to `config` win over it, so
`report.plan(config=cfg, group_rare=False)` changes just that one choice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

APPROVE = ("confident", "all", "safe")


@dataclass(frozen=True)
class CleanConfig:
    # planner phases (see flasheda.codegen.planner)
    drop_ids: bool = True
    drop_missing: bool = True
    drop_leaks: bool = True
    drop_null_target: bool = True
    fix_numeric_text: bool = True
    normalize_text: bool = True
    drop_duplicates: bool = True
    group_rare: bool = True
    missing_indicators: bool = True
    fill_nulls: bool = True
    fix_skew: bool = True
    clip_outliers: bool = False
    # safety
    approve: str = "confident"              # "confident" | "all" | "safe"
    protect: Tuple[str, ...] = ()           # never dropped or changed
    # formats
    decimal: Optional[str] = None           # None = detect, "." or ","
    date_formats: Dict[str, str] = field(default_factory=dict)
    # features
    features: bool = False

    def __post_init__(self):
        if self.approve not in APPROVE:
            raise ValueError(f"approve must be one of {APPROVE}")
        if self.decimal not in (None, ".", ","):
            raise ValueError("decimal must be None (detect), '.' or ','")
        protect = (self.protect,) if isinstance(self.protect, str) else tuple(self.protect)
        object.__setattr__(self, "protect", protect)
        object.__setattr__(self, "date_formats", dict(self.date_formats))

    def plan_options(self) -> Dict[str, Any]:
        """Keyword options for report.plan() (everything except `features`)."""
        opts = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "features"}
        if not opts["date_formats"]:
            opts.pop("date_formats")
        return opts

    def replace(self, **changes) -> "CleanConfig":
        from dataclasses import replace
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["protect"] = list(self.protect)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CleanConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown CleanConfig option(s): {unknown}")
        return cls(**data)

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CleanConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
