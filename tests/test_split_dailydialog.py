import json

from scripts.split_dailydialog import split_records


def test_split_dailydialog_keeps_dialogues_in_one_split(tmp_path) -> None:
    records = [
        {"id": f"dialogue-{dialogue}-pair-{pair}", "messages": []}
        for dialogue in range(10)
        for pair in range(2)
    ]
    source = tmp_path / "source.json"
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    source.write_text(json.dumps(records), encoding="utf-8")

    train_count, validation_count = split_records(
        source, train_path, validation_path, validation_ratio=0.2, seed=7
    )

    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    validation = [json.loads(line) for line in validation_path.read_text().splitlines()]
    train_dialogues = {record["id"].rsplit("-pair-", 1)[0] for record in train}
    validation_dialogues = {record["id"].rsplit("-pair-", 1)[0] for record in validation}
    assert train_count == 16
    assert validation_count == 4
    assert train_dialogues.isdisjoint(validation_dialogues)
