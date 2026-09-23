"""Score every scanned document and derive reproducible source weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_dataset.quality import quality_source_weights, score_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--prior", action="append", default=[], metavar="SOURCE=WEIGHT")
    args = parser.parse_args()
    priors = {item.split("=", 1)[0]: float(item.split("=", 1)[1]) for item in args.prior}
    documents, summaries = [], {}
    for path in args.paths:
        rows, summary = score_records(str(path), max_records=args.max_records)
        documents.extend(rows)
        source = path.parent.name
        summaries[source] = summary
    source_quality = {source: summary["mean_quality"] for source, summary in summaries.items()}
    weights = quality_source_weights(source_quality, priors or None)
    result = {"status": "passed", "sources": summaries, "quality_weights": weights, "protocol": "deterministic heuristic v1"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.scores_output:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        with args.scores_output.open("w", encoding="utf-8") as stream:
            for row in documents: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
if __name__ == "__main__": main()
