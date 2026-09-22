"""Deterministic end-to-end inference benchmark metrics."""
from __future__ import annotations

import statistics


def summarize_latency(samples):
    if not samples: raise ValueError("samples cannot be empty")
    ttft=[float(x["ttft_s"]) for x in samples]; itl=[float(x["itl_s"]) for x in samples if x.get("itl_s") is not None]; tps=[float(x["tokens"]) / max(float(x["duration_s"]),1e-9) for x in samples]
    return {"requests":len(samples),"ttft_p50":statistics.median(ttft),"ttft_p95":_pct(ttft,.95),"itl_p50":statistics.median(itl) if itl else 0.0,"tokens_per_second":sum(x["tokens"] for x in samples)/max(sum(x["duration_s"] for x in samples),1e-9),"request_throughput":len(samples)/max(sum(x["duration_s"] for x in samples),1e-9),"peak_memory_mb":max(float(x.get("peak_memory_mb",0)) for x in samples)}
def _pct(v,p):
    s=sorted(v); return s[round((len(s)-1)*p)]
