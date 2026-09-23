"""Audit dataset manifests for provenance, licensing, privacy, and allowed use."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
project_root = str(Path(__file__).resolve().parents[1])
src_directory = str(Path(project_root) / "src")
# ``datasets`` is also the name of the optional Hugging Face dependency.  A
# path added by an editable install may already exist later in ``sys.path``;
# move our source root ahead of site-packages rather than only adding it when
# absent.
sys.path[:] = [
    entry for entry in sys.path
    if str(Path(entry or ".").resolve()) not in {script_directory, src_directory}
]
sys.path.insert(0, src_directory)

from local_dataset.governance import (
    audit_dataset_files,
    audit_manifest_files,
    build_governance_report,
)
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--stage", help="override dataset_governance.stage from the training config")
    parser.add_argument(
        "--output", type=Path,
        help="also write the JSON result atomically (for example reports/data_quality.json)",
    )
    parser.add_argument(
        "--commercial-use", action=argparse.BooleanOptionalAction, default=None,
        help="override dataset_governance.commercial_use from the training config",
    )
    parser.add_argument(
        "--capability-audit", action="store_true",
        help="run the strict DATA-003 provenance, quality, contamination, and domain audit",
    )
    args = parser.parse_args()
    paths = list(args.paths)
    governance = {}
    capability_manifests = []
    expected_domains = {}
    if args.training_config:
        config = load_yaml(args.training_config)
        governance = config.get("dataset_governance") or {}
        paths.extend(config.get("train_files", []))
        paths.extend(config.get("validation_files", []))
        if args.capability_audit or governance.get("capability_audit", False):
            for source in (config.get("capability_corpus") or {}).get("sources", []):
                manifest = source.get("manifest") if isinstance(source, dict) else None
                domain = source.get("domain") if isinstance(source, dict) else None
                if manifest:
                    capability_manifests.append(Path(manifest))
                    expected_domains[Path(manifest)] = domain
    if not paths:
        parser.error("provide dataset paths or --training-config")
    strict = args.capability_audit or bool(governance.get("capability_audit", False))
    audit_kwargs = {
        "stage": args.stage or str(governance.get("stage", "training")),
        "commercial_use": (
            bool(args.commercial_use)
            if args.commercial_use is not None
            else bool(governance.get("commercial_use", False))
        ),
        "require_provenance": strict,
        "require_quality": strict,
        "require_contamination": strict,
        "required_domains": governance.get("required_domains", ()),
    }
    findings = audit_dataset_files(paths, **audit_kwargs)
    capability_findings = audit_manifest_files(
        capability_manifests,
        stage=audit_kwargs["stage"],
        commercial_use=audit_kwargs["commercial_use"],
        required_domains=audit_kwargs["required_domains"],
        expected_domains=expected_domains,
    ) if strict else []
    if capability_findings:
        findings.extend(capability_findings)
    result = (
        build_governance_report(paths + capability_manifests, **audit_kwargs)
        if strict
        else {
            "status": "failed" if findings else "passed",
            "datasets": len({str(Path(path).parent) for path in paths}),
            "findings": [
                {"path": str(item.path), "code": item.code, "message": item.message}
                for item in findings
            ],
        }
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(rendered)
            os.replace(temporary, args.output)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    print(rendered, end="")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
