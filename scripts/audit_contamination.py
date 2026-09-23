"""Audit train/evaluation exact and near-duplicate contamination independently."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory: sys.path.pop(0)
from local_dataset.contamination import audit_contamination, load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--max-records", type=int, default=100_000)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = load_documents(args.train, "train", args.max_records, args.shingle_size)
    evaluation = load_documents(args.evaluation, "evaluation", args.max_records, args.shingle_size)
    result = audit_contamination(train, evaluation, threshold=args.threshold)
    result["protocol"] = {"shingle_size": args.shingle_size, "max_records_per_split": args.max_records, "normalization": "NFKC + casefold + Unicode word tokens", "method": "SHA-256 exact match + deterministic MinHash band candidates + Jaccard"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(1 if result["status"] == "failed" else 0)
if __name__ == "__main__": main()
