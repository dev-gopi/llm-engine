"""Machine-readable dataset provenance and license policy checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


MANIFEST_NAME = "dataset-manifest.yaml"
VALID_POLICIES = {"off", "warn", "error"}
VALID_REVIEW_STATES = {"reviewed", "unreviewed"}
VALID_COMMERCIAL_USES = {"allowed", "prohibited", "unknown"}


@dataclass(frozen=True)
class GovernanceFinding:
    path: Path
    code: str
    message: str


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset manifest must contain a mapping: {source}")
    _validate_manifest(payload, source)
    return payload


def audit_dataset_files(
    paths: Iterable[str | Path],
    *,
    stage: str,
    commercial_use: bool = False,
) -> list[GovernanceFinding]:
    if not stage.strip():
        raise ValueError("dataset governance stage cannot be empty")
    findings: list[GovernanceFinding] = []
    checked_directories: set[Path] = set()
    for raw_path in paths:
        data_path = Path(raw_path)
        dataset_directory = data_path.parent
        if dataset_directory in checked_directories:
            continue
        checked_directories.add(dataset_directory)
        manifest_path = dataset_directory / MANIFEST_NAME
        if not manifest_path.is_file():
            findings.append(GovernanceFinding(
                data_path, "missing_manifest",
                f"no {MANIFEST_NAME} exists beside dataset {data_path}",
            ))
            continue
        try:
            manifest = load_dataset_manifest(manifest_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            findings.append(GovernanceFinding(data_path, "invalid_manifest", str(error)))
            continue
        license_info = manifest["license"]
        if license_info["review_status"] != "reviewed":
            findings.append(GovernanceFinding(
                data_path, "license_unreviewed",
                f"license review is incomplete for {manifest['name']}",
            ))
        if manifest["privacy_review"] != "reviewed":
            findings.append(GovernanceFinding(
                data_path, "privacy_unreviewed",
                f"privacy review is incomplete for {manifest['name']}",
            ))
        allowed_stages = set(manifest["allowed_stages"])
        if stage not in allowed_stages:
            findings.append(GovernanceFinding(
                data_path, "stage_not_allowed",
                f"{manifest['name']} is not approved for training stage {stage!r}",
            ))
        if commercial_use and license_info["commercial_use"] != "allowed":
            findings.append(GovernanceFinding(
                data_path, "commercial_use_not_allowed",
                f"{manifest['name']} is not approved for commercial use",
            ))
    return findings


def enforce_dataset_governance(
    paths: Iterable[str | Path],
    config: Mapping[str, Any] | None = None,
) -> list[GovernanceFinding]:
    settings = dict(config or {})
    policy = str(settings.get("policy", "warn")).lower()
    if policy not in VALID_POLICIES:
        raise ValueError("dataset governance policy must be off, warn, or error")
    if policy == "off":
        return []
    findings = audit_dataset_files(
        paths,
        stage=str(settings.get("stage", "training")),
        commercial_use=bool(settings.get("commercial_use", False)),
    )
    if findings and policy == "error":
        details = "\n".join(f"- [{item.code}] {item.message}" for item in findings)
        raise ValueError(f"dataset governance audit failed:\n{details}")
    return findings


def _validate_manifest(manifest: Mapping[str, Any], source: Path) -> None:
    required_strings = ("name", "source", "version")
    for key in required_strings:
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"{source}: {key} must be a non-empty string")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{source}: schema_version must be 1")
    license_info = manifest.get("license")
    if not isinstance(license_info, Mapping):
        raise ValueError(f"{source}: license must be a mapping")
    if not isinstance(license_info.get("identifier"), str) or not license_info["identifier"].strip():
        raise ValueError(f"{source}: license.identifier must be a non-empty string")
    if license_info.get("review_status") not in VALID_REVIEW_STATES:
        raise ValueError(f"{source}: license.review_status must be reviewed or unreviewed")
    if license_info.get("commercial_use") not in VALID_COMMERCIAL_USES:
        raise ValueError(
            f"{source}: license.commercial_use must be allowed, prohibited, or unknown"
        )
    stages = manifest.get("allowed_stages")
    if not isinstance(stages, list) or not stages or not all(
        isinstance(stage, str) and stage.strip() for stage in stages
    ):
        raise ValueError(f"{source}: allowed_stages must be a non-empty string list")
    if manifest.get("privacy_review") not in VALID_REVIEW_STATES:
        raise ValueError(f"{source}: privacy_review must be reviewed or unreviewed")
