"""FSDP-safe generation evaluation: rank-local generation, object aggregation, rank-0 metrics."""
from __future__ import annotations
import torch.distributed as dist

def evaluate_generation(model, records, generate_fn, *, context=None):
    local=list(records)
    if context is not None and context.world_size>1:
        local=[x for i,x in enumerate(local) if i % context.world_size == context.rank]
    results=[]
    was_training=model.training; model.eval()
    try:
        for record in local:
            results.append({"prompt":record["prompt"],"output":generate_fn(model,record)})
    finally:
        model.train(was_training)
    if context is not None and context.world_size>1 and dist.is_initialized():
        gathered=[None]*context.world_size; dist.all_gather_object(gathered,results)
        merged=[]
        for part in gathered: merged.extend(part or [])
        results=merged
    return results
