"""Context length ablation report builder."""
from __future__ import annotations


def compare_context_runs(runs):
    required=(512,1024,2048,4096,8192)
    normalized={int(r["context_length"]):r for r in runs}
    missing=[x for x in required if x not in normalized]
    return {"required_lengths":list(required),"missing":missing,"complete":not missing,"runs":normalized,"quality_deltas":{str(k): normalized[k].get("quality_delta") for k in normalized},"memory_mb":{str(k):normalized[k].get("peak_memory_mb") for k in normalized},"throughput":{str(k):normalized[k].get("tokens_per_second") for k in normalized}}
