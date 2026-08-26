"""Validate a model profile and estimate its size without allocating its weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from model.config import estimate_model_size, normalize_model_config
from utils.config import load_yaml


def _gib(value: int) -> float:
    return value / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_config", type=Path)
    args = parser.parse_args()
    config = normalize_model_config(load_yaml(args.model_config))
    size = estimate_model_size(config)
    print(json.dumps({
        "model_config": str(args.model_config),
        "parameters": size.parameters,
        "parameters_billions": round(size.parameters / 1e9, 3),
        "weights_fp32_gib": round(_gib(size.parameter_bytes_fp32), 3),
        "weights_bf16_fp16_gib": round(_gib(size.parameter_bytes_bf16), 3),
        "kv_cache_bf16_gib_per_max_length_sequence": round(
            _gib(size.kv_cache_bytes_bf16_per_sequence), 3
        ),
        "context_length": int(config["max_position"]),
        "note": "Training also needs gradients, optimizer states, activations, and temporary buffers.",
    }, indent=2))


if __name__ == "__main__":
    main()
