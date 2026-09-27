#!/usr/bin/env python3
"""Quantize a llama.cpp-compatible GGUF using the canonical llama-quantize tool."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

COMMON_QUANT_TYPES = frozenset(
    {
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_0",
        "Q4_1",
        "Q4_K_S",
        "Q4_K_M",
        "Q5_0",
        "Q5_1",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    }
)


def build_command(
    source: str | Path,
    output: str | Path,
    quant_type: str,
    *,
    executable: str = "llama-quantize",
) -> list[str]:
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"input GGUF not found: {source_path}")
    if source_path.suffix.lower() != ".gguf" or output_path.suffix.lower() != ".gguf":
        raise ValueError("input and output must use the .gguf extension")
    if source_path == output_path:
        raise ValueError("quantized output must differ from the source GGUF")
    normalized = quant_type.upper()
    if normalized not in COMMON_QUANT_TYPES:
        raise ValueError(
            f"unsupported quantization type {quant_type!r}; choose one of "
            + ", ".join(sorted(COMMON_QUANT_TYPES))
        )
    resolved = shutil.which(executable) or executable
    return [resolved, str(source_path), str(output_path), normalized]


def validate_gguf(path: str | Path) -> Path:
    artifact = Path(path)
    if not artifact.is_file() or artifact.stat().st_size < 24:
        raise RuntimeError(f"llama.cpp did not produce a valid GGUF: {artifact}")
    if artifact.read_bytes()[:4] != b"GGUF":
        raise RuntimeError(f"quantized artifact has invalid GGUF magic: {artifact}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--type", dest="quant_type", default="Q4_K_M")
    parser.add_argument("--llama-quantize", default="llama-quantize")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = build_command(
        args.source,
        args.output,
        args.quant_type,
        executable=args.llama_quantize,
    )
    if args.dry_run:
        print(subprocess.list2cmdline(command))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    validate_gguf(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
