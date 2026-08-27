import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from datasets.governance import (
    audit_dataset_files,
    enforce_dataset_governance,
    load_dataset_manifest,
)


def write_dataset(tmp_path, **overrides):
    directory = tmp_path / "example"
    directory.mkdir()
    data_path = directory / "train.jsonl"
    data_path.write_text(json.dumps({"text": "example"}) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "name": "Example",
        "source": "project/example",
        "version": "1",
        "license": {
            "identifier": "MIT",
            "review_status": "reviewed",
            "commercial_use": "allowed",
        },
        "allowed_stages": ["pretraining"],
        "privacy_review": "reviewed",
    }
    manifest.update(overrides)
    (directory / "dataset-manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return data_path, directory / "dataset-manifest.yaml"


def test_reviewed_manifest_passes_governance_audit(tmp_path) -> None:
    data_path, manifest_path = write_dataset(tmp_path)
    assert load_dataset_manifest(manifest_path)["name"] == "Example"
    assert audit_dataset_files(
        [data_path, data_path], stage="pretraining", commercial_use=True
    ) == []


def test_governance_reports_missing_unreviewed_and_disallowed_use(tmp_path) -> None:
    missing = tmp_path / "missing" / "train.jsonl"
    data_path, _ = write_dataset(
        tmp_path,
        license={
            "identifier": "LicenseRef-Unknown",
            "review_status": "unreviewed",
            "commercial_use": "unknown",
        },
        privacy_review="unreviewed",
    )
    findings = audit_dataset_files(
        [missing, data_path], stage="sft", commercial_use=True
    )
    assert {finding.code for finding in findings} == {
        "missing_manifest",
        "license_unreviewed",
        "privacy_unreviewed",
        "stage_not_allowed",
        "commercial_use_not_allowed",
    }


def test_strict_governance_policy_blocks_training_inputs(tmp_path) -> None:
    data_path = tmp_path / "train.jsonl"
    with pytest.raises(ValueError, match="dataset governance audit failed"):
        enforce_dataset_governance(
            [data_path], {"policy": "error", "stage": "pretraining"}
        )
    assert enforce_dataset_governance([data_path], {"policy": "off"}) == []


def test_manifest_schema_rejects_incomplete_license(tmp_path) -> None:
    _, manifest_path = write_dataset(tmp_path, license={"identifier": "MIT"})
    with pytest.raises(ValueError, match="license.review_status"):
        load_dataset_manifest(manifest_path)


def test_dataset_audit_cli_returns_machine_readable_failure(tmp_path) -> None:
    data_path = tmp_path / "unreviewed" / "train.jsonl"
    completed = subprocess.run(
        [sys.executable, "scripts/audit_datasets.py", str(data_path), "--stage", "pretraining"],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert payload["findings"][0]["code"] == "missing_manifest"
