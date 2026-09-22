"""Release manifests, model cards and dataset cards with reproducibility hashes."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: str | Path = ".") -> str | None:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class ReproducibilityManifest:
    source_revision: str | None
    python_version: str
    torch_version: str
    config_hashes: dict[str, str]
    tokenizer_hash: str | None
    dataset_hashes: dict[str, str]
    checkpoint_hash: str | None
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCard:
    model_name: str
    version: str
    architecture: str
    intended_use: str
    limitations: tuple[str, ...]
    training_data: tuple[str, ...]
    evaluation_summary: Mapping[str, Any]
    safety_summary: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def to_markdown(self) -> str:
        lines = [f"# {self.model_name}", "", f"Version: `{self.version}`", "", f"Architecture: `{self.architecture}`", "", "## Intended use", self.intended_use, "", "## Limitations"]
        lines.extend(f"- {item}" for item in self.limitations)
        lines.extend(["", "## Training data"])
        lines.extend(f"- {item}" for item in self.training_data)
        lines.extend(["", "## Evaluation", "```json", json.dumps(dict(self.evaluation_summary), indent=2, sort_keys=True), "```", "", "## Safety", "```json", json.dumps(dict(self.safety_summary), indent=2, sort_keys=True), "```", "", "## Provenance", "```json", json.dumps(dict(self.provenance), indent=2, sort_keys=True), "```"])
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class DatasetCard:
    name: str
    version: str
    sources: tuple[str, ...]
    license_notes: str
    preprocessing: tuple[str, ...]
    deduplication: str
    contamination_controls: tuple[str, ...]
    limitations: tuple[str, ...]
    hashes: Mapping[str, str]

    def to_markdown(self) -> str:
        lines=[f"# Dataset Card: {self.name}", "", f"Version: `{self.version}`", "", "## Sources"]
        lines.extend(f"- {x}" for x in self.sources)
        lines.extend(["", "## License", self.license_notes, "", "## Preprocessing"])
        lines.extend(f"- {x}" for x in self.preprocessing)
        lines.extend(["", "## Deduplication", self.deduplication, "", "## Contamination controls"])
        lines.extend(f"- {x}" for x in self.contamination_controls)
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {x}" for x in self.limitations)
        lines.extend(["", "## Artifact hashes", "```json", json.dumps(dict(self.hashes), indent=2, sort_keys=True), "```"])
        return "\n".join(lines) + "\n"


def build_reproducibility_manifest(
    *,
    root: str | Path = ".",
    config_paths: Sequence[str | Path] = (),
    tokenizer_path: str | Path | None = None,
    dataset_paths: Sequence[str | Path] = (),
    checkpoint_path: str | Path | None = None,
) -> ReproducibilityManifest:
    import sys

    import torch
    root = Path(root)
    config_hashes = {str(Path(path)): sha256_file(path) for path in config_paths if Path(path).is_file()}
    dataset_hashes = {str(Path(path)): sha256_file(path) for path in dataset_paths if Path(path).is_file()}
    tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path and Path(tokenizer_path).is_file() else None
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path and Path(checkpoint_path).is_file() else None
    environment = {key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES", "GOPI_DEVICE", "GOPI_MODEL_NAME") if key in os.environ}
    return ReproducibilityManifest(git_revision(root), sys.version, torch.__version__, config_hashes, tokenizer_hash, dataset_hashes, checkpoint_hash, environment)


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target=Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
