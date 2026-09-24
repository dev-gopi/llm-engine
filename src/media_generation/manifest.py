"""Reproducibility metadata for generated media artifacts."""
from __future__ import annotations
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any
import torch


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(*, kind: str, output: Path, prompt: str, negative_prompt: str, seed: int | None,
                   checkpoint: Path, config_path: Path, settings: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "created_at_unix": time.time(),
        "output": str(output),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "settings": settings,
        "elapsed_seconds": round(float(elapsed_seconds), 4),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
    }


def save_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
