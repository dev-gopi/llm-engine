#!/usr/bin/env python3
"""Write target-runtime qualification evidence for production-scale profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from runtime.qualification import qualify_profile, runtime_probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/runtime_qualification.json"))
    parser.add_argument("--strict", action="store_true", help="exit non-zero unless every profile qualifies")
    args = parser.parse_args()

    probes = runtime_probes()
    results = []
    for path in args.profiles:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            parser.error(f"profile must contain a YAML mapping: {path}")
        result = qualify_profile(payload, probes)
        result["source"] = str(path)
        results.append(result)

    report = {
        "schema_version": 1,
        "all_qualified": all(item["qualified"] for item in results),
        "profiles": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["all_qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
