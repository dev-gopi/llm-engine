import json

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_wikitext import iter_chunks, normalize_line, process_split


def test_wikitext_processing_groups_articles_and_normalizes_artifacts(tmp_path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    raw.mkdir()
    pq.write_table(
        pa.table({"text": [" = First article = \n", "Some @-@ text.\n", "", " = Second article = \n", "Another paragraph long enough.\n"]}),
        raw / "train-00000.parquet",
    )
    summary = process_split(raw, output, "train", min_characters=10)
    records = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["documents"] == 2
    assert records[0]["text"] == "= First article =\nSome-text."
    assert records[1]["source"] == "Salesforce/wikitext:wikitext-103-raw-v1"


def test_wikitext_line_normalization() -> None:
    assert normalize_line(" value @,@ next @.@ \n") == "value, next."


def test_document_chunking_preserves_all_content() -> None:
    text = "first paragraph\n" + "word " * 80 + "final tail"
    chunks = list(iter_chunks(text, 128))
    assert len(chunks) > 1
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())
    assert all(len(chunk) <= 128 for chunk in chunks)
