#!/usr/bin/env python3
"""Validate torchrun topology and execute a real cross-rank collective."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.multinode import validate_collectives


def tensor_parallel_smoke() -> float:
    import torch
    from inference.tensor_parallel import parallelize_minigpt
    from model.gpt import MiniGPT
    torch.manual_seed(17)
    model = MiniGPT(vocab_size=32, dim=8, layers=2, heads=2, kv_heads=2,
                    max_pos=16, position_type="none", ffn_hidden_dim=16).eval()
    for tensor in model.state_dict().values():
        torch.distributed.broadcast(tensor, src=0)
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(tokens)
        hidden = model.tok(tokens)
        expected_attention = model.blocks[0].attn(model.blocks[0].attention_norm(hidden))
        expected_ffn = model.blocks[0].ffn(model.blocks[0].ffn_norm(hidden))
        parallelize_minigpt(model)
        actual = model(tokens)
        attention_error = float((expected_attention - model.blocks[0].attn(model.blocks[0].attention_norm(hidden))).abs().max())
        ffn_error = float((expected_ffn - model.blocks[0].ffn(model.blocks[0].ffn_norm(hidden))).abs().max())
    error = float((expected - actual).abs().max())
    if error > 1e-5:
        raise RuntimeError(f"tensor-parallel logits differ by {error}; attention={attention_error}, ffn={ffn_error}")
    return error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"))
    parser.add_argument("--expected-nodes", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--tensor-parallel-smoke", action="store_true")
    args = parser.parse_args()
    report = validate_collectives(backend=args.backend, timeout_seconds=args.timeout_seconds)
    if args.tensor_parallel_smoke:
        report["tensor_parallel_max_error"] = tensor_parallel_smoke()
    if args.expected_nodes is not None and report["observed_nodes"] != args.expected_nodes:
        raise SystemExit(
            f"expected {args.expected_nodes} hosts but torchrun reached {report['observed_nodes']}: {report['hosts']}"
        )
    if report["rank"] == 0:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
