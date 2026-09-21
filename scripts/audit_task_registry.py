#!/usr/bin/env python3
"""Validate stable task IDs and dependencies in docs/TASKS.md."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HEADING_RE = re.compile(r"^###\s+([A-Z][A-Z0-9-]+):\s*(.+?)\s*$", re.MULTILINE)
ID_RE = re.compile(r"^[-*]\s+\*\*ID\*\*:\s+`([A-Z][A-Z0-9-]+)`\s*$", re.MULTILINE)
DEP_RE = re.compile(r"^[-*]\s+\*\*Dependencies\*\*:\s*(.+?)\s*$", re.MULTILINE)
TOKEN_RE = re.compile(r"`([A-Z][A-Z0-9-]+)`")


def audit_task_registry(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    headings = list(HEADING_RE.finditer(text))
    ids = [match.group(1) for match in headings]
    counts = Counter(ids)
    duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
    known = set(ids)
    mismatched_ids: list[dict[str, str]] = []
    undefined_dependencies: set[str] = set()

    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[start:end]
        declared = ID_RE.search(section)
        if declared and declared.group(1) != heading.group(1):
            mismatched_ids.append({"heading": heading.group(1), "declared": declared.group(1)})
        dependency = DEP_RE.search(section)
        if dependency:
            for task_id in TOKEN_RE.findall(dependency.group(1)):
                if task_id not in known:
                    undefined_dependencies.add(task_id)

    return {
        "path": str(path),
        "sections": len(headings),
        "unique_ids": len(known),
        "duplicate_ids": duplicates,
        "mismatched_ids": mismatched_ids,
        "undefined_dependencies": sorted(undefined_dependencies),
        "passed": not duplicates and not mismatched_ids and not undefined_dependencies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path("docs/TASKS.md"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_task_registry(args.path)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
