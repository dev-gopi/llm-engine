"""Machine-readable dataset provenance, quality, contamination, and license checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MANIFEST_NAME = "dataset-manifest.yaml"
VALID_POLICIES = {"off", "warn", "error"}
VALID_REVIEW_STATES = {"reviewed", "unreviewed"}
VALID_COMMERCIAL_USES = {"allowed", "prohibited", "unknown"}
VALID_AUDIT_STATES = {"passed", "failed", "not_run"}
CAPABILITY_DOMAINS = {
    "web", "documentation", "code", "mathematics", "science",
    "multilingual", "conversation", "instruction", "reasoning", "tool-use",
}


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
    require_provenance: bool = False,
    require_quality: bool = False,
    require_contamination: bool = False,
    required_domains: Iterable[str] = (),
) -> list[GovernanceFinding]:
    """Audit dataset manifests without loading corpus records into memory.

    The optional strict checks are used by DATA-003. Existing DATA-002 callers
    retain the original license/privacy/stage behavior by leaving them false.
    """
    if not stage.strip():
        raise ValueError("dataset governance stage cannot be empty")
    required_domain_set = {str(domain).strip() for domain in required_domains if str(domain).strip()}
    invalid_domains = required_domain_set - CAPABILITY_DOMAINS
    if invalid_domains:
        raise ValueError(f"unknown capability domains: {sorted(invalid_domains)}")

    findings: list[GovernanceFinding] = []
    checked_directories: set[Path] = set()
    seen_domains: set[str] = set()
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

        domain = manifest.get("domain")
        if domain:
            seen_domains.add(str(domain))
        elif required_domain_set or require_provenance:
            findings.append(GovernanceFinding(
                data_path, "domain_missing",
                f"{manifest['name']} has no capability domain",
            ))

        if require_provenance:
            provenance = manifest.get("provenance")
            if not isinstance(provenance, Mapping):
                findings.append(GovernanceFinding(
                    data_path, "provenance_missing",
                    f"{manifest['name']} has no provenance audit block",
                ))
            else:
                for key in ("upstream", "acquisition", "acquired_at", "content_sha256"):
                    if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                        findings.append(GovernanceFinding(
                            data_path, f"provenance_{key}_missing",
                            f"{manifest['name']} provenance.{key} must be non-empty",
                        ))

        if require_quality:
            quality = manifest.get("quality")
            if not isinstance(quality, Mapping):
                findings.append(GovernanceFinding(
                    data_path, "quality_audit_missing",
                    f"{manifest['name']} has no quality audit block",
                ))
            else:
                if quality.get("status") != "passed":
                    findings.append(GovernanceFinding(
                        data_path, "quality_audit_not_passed",
                        f"{manifest['name']} quality audit status is {quality.get('status')!r}",
                    ))
                if not isinstance(quality.get("metrics"), Mapping) or not quality["metrics"]:
                    findings.append(GovernanceFinding(
                        data_path, "quality_metrics_missing",
                        f"{manifest['name']} must publish numeric quality metrics",
                    ))
                elif any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in quality["metrics"].values()
                ):
                    findings.append(GovernanceFinding(
                        data_path, "quality_metrics_invalid",
                        f"{manifest['name']} quality.metrics must contain only numbers",
                    ))

        if require_contamination:
            contamination = manifest.get("contamination_audit")
            if not isinstance(contamination, Mapping):
                findings.append(GovernanceFinding(
                    data_path, "contamination_audit_missing",
                    f"{manifest['name']} has no contamination audit block",
                ))
            else:
                if contamination.get("status") != "passed":
                    findings.append(GovernanceFinding(
                        data_path, "contamination_audit_not_passed",
                        f"{manifest['name']} contamination audit status is {contamination.get('status')!r}",
                    ))
                for key in ("method", "audited_at"):
                    if not isinstance(contamination.get(key), str) or not contamination[key].strip():
                        findings.append(GovernanceFinding(
                            data_path, f"contamination_{key}_missing",
                            f"{manifest['name']} contamination_audit.{key} must be non-empty",
                        ))
                if not isinstance(contamination.get("overlap_count"), int) or contamination["overlap_count"] < 0:
                    findings.append(GovernanceFinding(
                        data_path, "contamination_overlap_invalid",
                        f"{manifest['name']} contamination_audit.overlap_count must be a non-negative integer",
                    ))
                if contamination.get("overlap_count") != 0:
                    findings.append(GovernanceFinding(
                        data_path, "contamination_detected",
                        f"{manifest['name']} has {contamination.get('overlap_count')} benchmark contamination overlaps",
                    ))

    if required_domain_set:
        missing_domains = sorted(required_domain_set - seen_domains)
        for domain in missing_domains:
            findings.append(GovernanceFinding(
                Path("<capability-corpus>"), "required_domain_missing",
                f"no audited dataset declares required capability domain {domain!r}",
            ))
    return findings


def build_governance_report(
    paths: Iterable[str | Path],
    *,
    stage: str,
    commercial_use: bool = False,
    require_provenance: bool = False,
    require_quality: bool = False,
    require_contamination: bool = False,
    required_domains: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a deterministic, publishable governance report for dataset manifests."""
    unique_paths = list(dict.fromkeys(Path(path) for path in paths))
    findings = audit_dataset_files(
        unique_paths,
        stage=stage,
        commercial_use=commercial_use,
        require_provenance=require_provenance,
        require_quality=require_quality,
        require_contamination=require_contamination,
        required_domains=required_domains,
    )
    manifests = []
    for path in unique_paths:
        manifest_path = path.parent / MANIFEST_NAME
        if manifest_path.is_file():
            try:
                manifest = load_dataset_manifest(manifest_path)
                manifests.append({
                    "path": str(path),
                    "manifest": str(manifest_path),
                    "name": manifest["name"],
                    "source": manifest["source"],
                    "version": manifest["version"],
                    "domain": manifest.get("domain"),
                    "license": manifest["license"],
                    "quality": manifest.get("quality"),
                    "contamination_audit": manifest.get("contamination_audit"),
                })
            except (OSError, ValueError, yaml.YAMLError):
                pass
    return {
        "schema_version": 1,
        "audit": "DATA-003 capability corpus governance",
        "status": "passed" if not findings else "blocked",
        "activation": "allowed" if not findings else "blocked",
        "stage": stage,
        "commercial_use": commercial_use,
        "required_domains": sorted(set(required_domains)),
        "datasets": manifests,
        "findings": [
            {"path": str(item.path), "code": item.code, "message": item.message}
            for item in findings
        ],
    }


def audit_manifest_files(
    manifests: Iterable[str | Path],
    *,
    stage: str,
    commercial_use: bool = False,
    required_domains: Iterable[str] = (),
    expected_domains: Mapping[str | Path, str] | None = None,
) -> list[GovernanceFinding]:
    """Audit capability-corpus manifest files before they are activated."""
    findings: list[GovernanceFinding] = []
    expected = {str(Path(path)): str(domain) for path, domain in (expected_domains or {}).items()}
    seen_domains: set[str] = set()
    for raw_path in dict.fromkeys(Path(path) for path in manifests):
        if not raw_path.is_file():
            findings.append(GovernanceFinding(raw_path, "missing_manifest", f"dataset manifest not found: {raw_path}"))
            continue
        try:
            manifest = load_dataset_manifest(raw_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            findings.append(GovernanceFinding(raw_path, "invalid_manifest", str(error)))
            continue
        domain = manifest.get("domain")
        if not isinstance(domain, str) or domain not in CAPABILITY_DOMAINS:
            findings.append(GovernanceFinding(raw_path, "domain_missing", f"{manifest['name']} has no valid capability domain"))
        else:
            seen_domains.add(domain)
            expected_domain = expected.get(str(raw_path))
            if expected_domain and domain != expected_domain:
                findings.append(GovernanceFinding(
                    raw_path, "domain_mismatch",
                    f"{manifest['name']} declares domain {domain!r}, expected {expected_domain!r}",
                ))
        license_info = manifest["license"]
        if license_info["review_status"] != "reviewed":
            findings.append(GovernanceFinding(raw_path, "license_unreviewed", f"license review is incomplete for {manifest['name']}"))
        if manifest["privacy_review"] != "reviewed":
            findings.append(GovernanceFinding(raw_path, "privacy_unreviewed", f"privacy review is incomplete for {manifest['name']}"))
        if stage not in set(manifest["allowed_stages"]):
            findings.append(GovernanceFinding(raw_path, "stage_not_allowed", f"{manifest['name']} is not approved for training stage {stage!r}"))
        if commercial_use and license_info["commercial_use"] != "allowed":
            findings.append(GovernanceFinding(raw_path, "commercial_use_not_allowed", f"{manifest['name']} is not approved for commercial use"))

        provenance = manifest.get("provenance")
        if not isinstance(provenance, Mapping):
            findings.append(GovernanceFinding(raw_path, "provenance_missing", f"{manifest['name']} has no provenance audit block"))
        else:
            for key in ("upstream", "acquisition", "acquired_at", "content_sha256"):
                if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                    findings.append(GovernanceFinding(raw_path, f"provenance_{key}_missing", f"{manifest['name']} provenance.{key} must be non-empty"))
            digest = provenance.get("content_sha256", "")
            if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower())):
                findings.append(GovernanceFinding(raw_path, "provenance_hash_invalid", f"{manifest['name']} provenance.content_sha256 must be a SHA-256 hex digest"))

        quality = manifest.get("quality")
        if not isinstance(quality, Mapping):
            findings.append(GovernanceFinding(raw_path, "quality_audit_missing", f"{manifest['name']} has no quality audit block"))
        else:
            if quality.get("status") != "passed":
                findings.append(GovernanceFinding(raw_path, "quality_audit_not_passed", f"{manifest['name']} quality audit status is {quality.get('status')!r}"))
            metrics = quality.get("metrics")
            if not isinstance(metrics, Mapping) or not metrics:
                findings.append(GovernanceFinding(raw_path, "quality_metrics_missing", f"{manifest['name']} must publish numeric quality metrics"))
            elif any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in metrics.values()):
                findings.append(GovernanceFinding(raw_path, "quality_metrics_invalid", f"{manifest['name']} quality.metrics must contain only numbers"))

        contamination = manifest.get("contamination_audit")
        if not isinstance(contamination, Mapping):
            findings.append(GovernanceFinding(raw_path, "contamination_audit_missing", f"{manifest['name']} has no contamination audit block"))
        else:
            if contamination.get("status") != "passed":
                findings.append(GovernanceFinding(raw_path, "contamination_audit_not_passed", f"{manifest['name']} contamination audit status is {contamination.get('status')!r}"))
            for key in ("method", "audited_at"):
                if not isinstance(contamination.get(key), str) or not contamination[key].strip():
                    findings.append(GovernanceFinding(raw_path, f"contamination_{key}_missing", f"{manifest['name']} contamination_audit.{key} must be non-empty"))
            overlap = contamination.get("overlap_count")
            if not isinstance(overlap, int) or overlap < 0:
                findings.append(GovernanceFinding(raw_path, "contamination_overlap_invalid", f"{manifest['name']} contamination_audit.overlap_count must be a non-negative integer"))
            elif overlap:
                findings.append(GovernanceFinding(raw_path, "contamination_detected", f"{manifest['name']} has {overlap} benchmark contamination overlaps"))

    missing_domains = sorted(set(required_domains) - seen_domains)
    for domain in missing_domains:
        findings.append(GovernanceFinding(Path("<capability-corpus>"), "required_domain_missing", f"no audited manifest declares required capability domain {domain!r}"))
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
        require_provenance=bool(settings.get("require_provenance", False)),
        require_quality=bool(settings.get("require_quality", False)),
        require_contamination=bool(settings.get("require_contamination", False)),
        required_domains=settings.get("required_domains", ()),
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
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError(f"{source}: schema_version must be 1 or 2")
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
    if manifest.get("schema_version") == 2:
        if not isinstance(manifest.get("domain"), str) or manifest["domain"] not in CAPABILITY_DOMAINS:
            raise ValueError(f"{source}: schema_version 2 requires a valid capability domain")
        provenance = manifest.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{source}: schema_version 2 requires provenance")
        quality = manifest.get("quality")
        if not isinstance(quality, Mapping):
            raise ValueError(f"{source}: schema_version 2 requires quality")
        contamination = manifest.get("contamination_audit")
        if not isinstance(contamination, Mapping):
            raise ValueError(f"{source}: schema_version 2 requires contamination_audit")
