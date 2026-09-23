"""Compare recorded corpus-mixture experiments against capability metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_dataset.mixture_ablation import MixtureObservation, compare_mixtures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON array of recorded experiment observations")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, list): raise ValueError("input must contain a JSON array or JSONL records")
    observations = [MixtureObservation(str(item["name"]), item["mixture"], item["metrics"]) for item in raw]
    result = compare_mixtures(observations, args.baseline)
    result["protocol"] = "recorded checkpoint/benchmark observations; descriptive deltas only"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
if __name__ == "__main__": main()
