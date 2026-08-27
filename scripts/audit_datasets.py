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
    parser.add_argument("--stage", default="training")
    parser.add_argument("--commercial-use", action="store_true")
    args = parser.parse_args()
    paths = list(args.paths)
    if args.training_config:
        config = load_yaml(args.training_config)
        paths.extend(config.get("train_files", []))
        paths.extend(config.get("validation_files", []))
    if not paths:
        parser.error("provide dataset paths or --training-config")
    findings = audit_dataset_files(
        paths, stage=args.stage, commercial_use=args.commercial_use,
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
