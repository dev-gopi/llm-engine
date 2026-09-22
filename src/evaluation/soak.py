"""Serving soak-test result contract."""
from __future__ import annotations


def evaluate_soak(samples, *, max_error_rate=0.01, max_memory_growth_mb=128.0):
    if not samples: raise ValueError("samples cannot be empty")
    errors=sum(1 for x in samples if not x.get("ok",False)); mem=[float(x.get("memory_mb",0)) for x in samples]
    growth=max(mem)-min(mem)
    return {"requests":len(samples),"error_rate":errors/len(samples),"memory_growth_mb":growth,"passed":errors/len(samples)<=max_error_rate and growth<=max_memory_growth_mb}
