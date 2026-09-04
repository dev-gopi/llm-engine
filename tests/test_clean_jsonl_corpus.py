import json
import subprocess
import sys
from pathlib import Path

from datasets.filters import CorpusFilter
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def _tokenizer(path: Path) -> None:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS}).save(path)


def test_cleaner_redacts_and_deduplicates_against_validation(tmp_path) -> None:
    tokenizer_path = tmp_path / "tokenizer"
    _tokenizer(tokenizer_path)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    clean = tmp_path / "clean.jsonl"
    repeated = "This sufficiently long training example belongs to Jane at jane@example.com."
    held_out = "This sufficiently long sentence must remain exclusive to validation data."
    train.write_text("\n".join(json.dumps({"text": text}) for text in (held_out, repeated, repeated)) + "\n")
    validation.write_text(json.dumps({"text": held_out}) + "\n")

    subprocess.run([
        sys.executable, "scripts/clean_jsonl_corpus.py", str(train),
        "--output", str(clean), "--tokenizer", str(tokenizer_path),
        "--exclude", str(validation), "--near-duplicate-distance", "0",
    ], check=True)

    rows = [json.loads(line) for line in clean.read_text().splitlines()]
    audit = json.loads(clean.with_suffix(".audit.json").read_text())
    assert len(rows) == 1
    assert "<email>" in rows[0]["text"]
    assert rows[0]["language"] == "en"
    assert audit["filter_stats"]["contamination"] == 1
    assert audit["filter_stats"]["duplicate"] == 1
    assert audit["output_records"] == 1


def test_filter_can_preserve_code_indentation() -> None:
    source = "def example():\n    value = 1\n    return value"
    corpus_filter = CorpusFilter(min_chars=5, preserve_whitespace=True)
    assert corpus_filter.apply(source) == source
