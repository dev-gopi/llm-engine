"""Programmatic verification adapters for reasoning outputs."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    kind: str
    expected: object | None
    observed: object | None
    error: str | None = None

def verify_math(answer: str, expected: float, *, tolerance=1e-9):
    nums=re.findall(r"[-+]?\d+(?:\.\d+)?", answer.replace(",",""))
    observed=float(nums[-1]) if nums else None
    return VerificationResult(observed is not None and math.isclose(observed,float(expected),rel_tol=tolerance,abs_tol=tolerance),"math",expected,observed,None if observed is not None else "no numeric answer")

def verify_code(output: str, expected: str):
    return VerificationResult(output.strip()==expected.strip(),"code",expected,output.strip())

def verify_logic(answer: str, expected: str):
    return VerificationResult(answer.strip().lower()==expected.strip().lower(),"logic",expected,answer.strip())

def verify_reasoning(record):
    kind=record.get("kind","logic")
    if kind=="math": return verify_math(str(record.get("answer","")), record["expected"])
    if kind=="code": return verify_code(str(record.get("answer","")), str(record.get("expected","")))
    return verify_logic(str(record.get("answer","")), str(record.get("expected","")))
