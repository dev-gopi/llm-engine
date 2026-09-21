"""Reusable OpenAI-compatible API conformance runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ConformanceCase:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ConformanceReport:
    suite: str
    cases: tuple[ConformanceCase, ...]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(case.name for case in self.cases if not case.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "passed": self.passed,
            "cases": [case.__dict__ for case in self.cases],
            "failures": list(self.failures),
        }


def run_case(name: str, function: Callable[[], Any]) -> ConformanceCase:
    try:
        result = function()
        if result is False:
            return ConformanceCase(name, False, "assertion returned false")
        return ConformanceCase(name, True)
    except Exception as error:
        return ConformanceCase(name, False, f"{type(error).__name__}: {error}")
