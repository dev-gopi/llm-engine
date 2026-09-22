#!/usr/bin/env python3
"""Launch llama.cpp's ``llama-server`` with a GGUF model using safe arguments."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def command(args: argparse.Namespace) -> list[str]:
    executable = shutil.which(args.llama_server) or args.llama_server
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"GGUF model not found: {model}")
    if model.suffix.lower() != ".gguf":
        raise ValueError("model must be a .gguf file")
    if args.context < 1 or args.parallel < 1 or args.gpu_layers < 0:
        raise ValueError("context/parallel must be positive and gpu-layers non-negative")
    result = [
        executable, "--model", str(model), "--host", args.host, "--port", str(args.port),
        "--ctx-size", str(args.context), "--parallel", str(args.parallel),
        "--n-gpu-layers", str(args.gpu_layers),
    ]
    if args.flash_attention:
        result.append("--flash-attn")
    if args.cache_type_k:
        result += ["--cache-type-k", args.cache_type_k]
    if args.cache_type_v:
        result += ["--cache-type-v", args.cache_type_v]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to the GGUF model (for example model.gguf)")
    parser.add_argument("--llama-server", default="llama-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--flash-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cmd = command(args)
    if args.dry_run:
        print(subprocess.list2cmdline(cmd))
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
