import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import prepare_hf_dataset


def test_none_sizes_process_complete_dataset(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"instruction": f"Question {index}", "output": f"Answer {index}"}
        for index in range(40)
    ]), source)
    monkeypatch.setattr(
        prepare_hf_dataset, "fetch_hf_parquet_urls",
        lambda *_args, **_kwargs: [source.as_uri()],
    )
    output = tmp_path / "processed"

    counts = prepare_hf_dataset.download_and_convert_dataset(
        "test/full", output, train_size=None, validation_size=None, test_size=None,
    )

    assert sum(counts.values()) == 40
    assert all((output / f"{split}.jsonl").is_file() for split in counts)
    records = [json.loads(line) for split in counts for line in (output / f"{split}.jsonl").read_text().splitlines()]
    assert len(records) == 40


def test_bounded_mode_requires_all_split_sizes(tmp_path) -> None:
    with pytest.raises(ValueError, match="all three split sizes"):
        prepare_hf_dataset.download_and_convert_dataset(
            "test/partial", tmp_path, train_size=10, validation_size=None,
        )


def test_preference_scores_are_preserved(tmp_path, monkeypatch) -> None:
    source = tmp_path / "helpsteer.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "prompt": "How can I help?",
        "response": "Give a clear answer.",
        "helpfulness": 4,
        "correctness": 3,
        "coherence": 4,
        "complexity": 2,
        "verbosity": 1,
    }]), source)
    monkeypatch.setattr(
        prepare_hf_dataset, "fetch_hf_parquet_urls",
        lambda *_args, **_kwargs: [source.as_uri()],
    )
    output = tmp_path / "processed"

    prepare_hf_dataset.download_and_convert_dataset(
        "nvidia/HelpSteer", output,
        train_size=1, validation_size=0, test_size=0,
    )

    record = json.loads((output / "train.jsonl").read_text().splitlines()[0])
    assert record["scores"] == {
        "helpfulness": 4,
        "correctness": 3,
        "coherence": 4,
        "complexity": 2,
        "verbosity": 1,
    }


def test_full_preference_dataset_keeps_prompt_candidates_together(
    tmp_path, monkeypatch,
) -> None:
    source = tmp_path / "helpsteer.parquet"
    pq.write_table(pa.Table.from_pylist([
        {
            "prompt": "Shared prompt",
            "response": f"Candidate {index}",
            "helpfulness": index,
        }
        for index in range(5)
    ]), source)
    monkeypatch.setattr(
        prepare_hf_dataset, "fetch_hf_parquet_urls",
        lambda *_args, **_kwargs: [source.as_uri()],
    )
    output = tmp_path / "processed"

    prepare_hf_dataset.download_and_convert_dataset(
        "nvidia/HelpSteer", output,
        train_size=None, validation_size=None, test_size=None,
    )

    populated_splits = [
        split
        for split in ("train", "validation", "test")
        if (output / f"{split}.jsonl").read_text().strip()
    ]
    assert len(populated_splits) == 1
    assert len((output / f"{populated_splits[0]}.jsonl").read_text().splitlines()) == 5
