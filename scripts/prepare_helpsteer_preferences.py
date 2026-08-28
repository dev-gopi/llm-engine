"""Build deterministic DPO preference pairs from processed HelpSteer records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SCORE_FIELDS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")


def quality(record: dict) -> float:
    scores = record.get("scores", {})
    values = [float(scores.get(field, 0)) for field in SCORE_FIELDS]
    return sum(values) / len(values)


def extract_pair(record: dict) -> tuple[str, str] | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    prompt = next((message.get("content", "").strip() for message in messages if message.get("role") == "user"), "")
    answer = next((message.get("content", "").strip() for message in reversed(messages) if message.get("role") == "assistant"), "")
    return (prompt, answer) if prompt and answer else None


def convert(source: Path) -> list[dict]:
    groups: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            pair = extract_pair(record)
            if pair is not None:
                prompt, answer = pair
                groups[prompt].append((quality(record), answer))

    preferences = []
    for index, prompt in enumerate(sorted(groups)):
        candidates = sorted(groups[prompt], key=lambda item: (item[0], item[1]))
        if len(candidates) < 2 or candidates[0][1] == candidates[-1][1] or candidates[0][0] == candidates[-1][0]:
            continue
        preferences.append({
            "id": f"helpsteer-preference-{index}",
            "source": "nvidia/HelpSteer",
            "prompt": prompt,
            "chosen": candidates[-1][1],
            "rejected": candidates[0][1],
        })
    return preferences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/helpsteer"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/preferences"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "validation"):
        records = convert(args.input_dir / f"{split}.jsonl")
        destination = args.output_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = len(records)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
