"""Create leakage-resistant DailyDialog train and validation JSONL splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split_records(
    source: Path,
    train_output: Path,
    validation_output: Path,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[int, int]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between zero and one")
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("source must contain at least two records")
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        identifier = str(record.get("id", ""))
        dialogue_id = identifier.rsplit("-pair-", 1)[0]
        if not dialogue_id:
            raise ValueError("every record must have a dialogue-based id")
        groups[dialogue_id].append(record)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    validation_groups = set(group_ids[:max(1, round(len(group_ids) * validation_ratio))])
    train = [record for group in group_ids if group not in validation_groups for record in groups[group]]
    validation = [record for group in group_ids if group in validation_groups for record in groups[group]]
    for path, split in ((train_output, train), (validation_output, validation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in split),
            encoding="utf-8",
        )
    return len(train), len(validation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/dailydialog/dailydialog-conversations.json"))
    parser.add_argument("--train-output", type=Path, default=Path("data/processed/dailydialog/train.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/processed/dailydialog/validation.jsonl"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train, validation = split_records(
        args.source, args.train_output, args.validation_output,
        validation_ratio=args.validation_ratio, seed=args.seed,
    )
    print(json.dumps({"train": train, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
