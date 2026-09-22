"""Protected-capability release regression gate."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateRule:
    name:str; direction:str="max"; max_regression:float=0.0; required:bool=True

class ReleaseGate:
    def __init__(self,rules):
        self.rules=tuple(rules)
        if any(r.direction not in {"max","min"} for r in self.rules): raise ValueError("invalid gate direction")
    def evaluate(self, baseline, candidate):
        failures=[]; details={}
        for r in self.rules:
            if r.name not in candidate or r.name not in baseline:
                if r.required: failures.append(f"missing:{r.name}")
                continue
            b=float(baseline[r.name]); c=float(candidate[r.name])
            delta=(c-b) if r.direction=="max" else (b-c)
            allowed=abs(b)*r.max_regression
            details[r.name]={"baseline":b,"candidate":c,"delta":delta,"allowed":allowed}
            if delta < -allowed: failures.append(f"regression:{r.name}")
        return {"passed":not failures,"failures":failures,"details":details}
