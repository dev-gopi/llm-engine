"""Versioned external benchmark adapters with deterministic scoring."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable
import hashlib, json

@dataclass(frozen=True)
class BenchmarkSpec:
    name: str; version: str; dataset_hash: str; metric: str
    def fingerprint(self): return hashlib.sha256(json.dumps(self.__dict__,sort_keys=True).encode()).hexdigest()

@dataclass(frozen=True)
class BenchmarkResult:
    spec: BenchmarkSpec; count: int; score: float; controls: dict

class BenchmarkAdapter:
    def __init__(self,spec:BenchmarkSpec, scorer:Callable[[object,object],float]): self.spec=spec; self.scorer=scorer
    def run(self, pairs:Iterable[tuple[object,object]], *, controls=None):
        values=[float(self.scorer(a,b)) for a,b in pairs]
        return BenchmarkResult(self.spec,len(values),sum(values)/len(values) if values else 0.0,dict(controls or {}))
