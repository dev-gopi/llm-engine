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
