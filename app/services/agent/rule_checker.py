"""
Rule-based sanity check — placeholder, always passes for now.
Fill in validation logic later when rules are decided.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    ok: bool
    issues: list[str] = field(default_factory=list)


def sanity_check(query: str, schema_tables: list[dict], db_type: str) -> CheckResult:
    return CheckResult(ok=True)
