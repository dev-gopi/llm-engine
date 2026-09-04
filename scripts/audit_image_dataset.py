"""Audit image readability, dimensions, size, and exact duplicates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from image_data.audit import audit_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--fail-on-corrupt", action="store_true")
    args = parser.parse_args()
    audit = audit_images(args.data)
    print(json.dumps(audit.to_dict(), indent=2))
    if args.fail_on_corrupt and audit.corrupt:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
