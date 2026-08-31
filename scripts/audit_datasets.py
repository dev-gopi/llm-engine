"""Audit dataset manifests for provenance, licensing, privacy, and allowed use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from datasets.governance import audit_dataset_files
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--stage", help="override dataset_governance.stage from the training config")
    parser.add_argument(
        "--commercial-use", action=argparse.BooleanOptionalAction, default=None,
        help="override dataset_governance.commercial_use from the training config",
    )
    args = parser.parse_args()
    paths = list(args.paths)
    governance = {}
    if args.training_config:
        config = load_yaml(args.training_config)
        governance = config.get("dataset_governance") or {}
        paths.extend(config.get("train_files", []))
        paths.extend(config.get("validation_files", []))
    if not paths:
        parser.error("provide dataset paths or --training-config")
    findings = audit_dataset_files(
        paths,
        stage=args.stage or str(governance.get("stage", "training")),
        commercial_use=(
            bool(args.commercial_use)
            if args.commercial_use is not None
            else bool(governance.get("commercial_use", False))
        ),
    )
    result = {
        "status": "failed" if findings else "passed",
        "datasets": len({str(Path(path).parent) for path in paths}),
        "findings": [
            {"path": str(item.path), "code": item.code, "message": item.message}
            for item in findings
        ],
    }
    print(json.dumps(result, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
