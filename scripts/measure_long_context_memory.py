"""Measure checkpoint-backed peak memory across CTX-002 context profiles."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model.gpt import MiniGPT
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lengths = [int(v) for v in args.lengths.split(",") if v.strip()]
    if not lengths or any(v < 128 for v in lengths):
        parser.error("lengths must contain positive values >= 128")
    device = resolve_device(args.device)
    cfg = load_yaml(args.model_config)
    model = MiniGPT.from_config(cfg, device=device)
    load_checkpoint(args.checkpoint, model, map_location=device, restore_rng=False, strict=True)
    model.eval()
    results=[]
    for length in lengths:
        if length > model.max_positions:
            results.append({"context_length": length, "status": "blocked", "reason": "exceeds_model_max_position"})
            continue
        tokens = torch.zeros((1, length), dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        started=time.perf_counter()
        try:
            with torch.inference_mode():
                model(tokens, logits_to_keep=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            results.append({
                "context_length": length, "status": "passed",
                "elapsed_seconds": time.perf_counter()-started,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "allocated_memory_bytes": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None,
            })
        except torch.cuda.OutOfMemoryError:
            results.append({"context_length": length, "status": "oom", "elapsed_seconds": time.perf_counter()-started})
            torch.cuda.empty_cache()
    report={"schema_version":1,"task":"CTX-002","checkpoint":str(args.checkpoint),"model_config":str(args.model_config),"device":str(device),"results":results,"claim_status":"validated_only_if_all_requested_profiles_pass"}
    rendered=json.dumps(report, indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

if __name__ == "__main__":
    main()
