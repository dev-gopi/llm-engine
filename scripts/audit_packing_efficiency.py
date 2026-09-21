"""Measure packed-token and dynamic-padding efficiency from JSONL records."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datasets.packing import packing_efficiency, batch_padding_efficiency

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pad-to-multiple-of", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = []
    with args.path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                lengths.append(int(row.get("token_count", len(row.get("token_ids", [])) + 1)))
    result = {"path": str(args.path), "capacity": args.capacity,
              "packed": packing_efficiency(lengths, args.capacity),
              "dynamic_batch_padding": batch_padding_efficiency(lengths, args.batch_size, args.pad_to_multiple_of),
              "protocol": "token_count includes loader BOS; useful-token utilization excludes padding"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
