import json
import subprocess
import sys
from pathlib import Path

import pytest
from datasets.loader import TextDataset, LazyJSONLDataset
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def _tokenizer(path: Path) -> None:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS}).save(path)


def test_packer_bounds_rows_and_keeps_eos_boundaries(tmp_path) -> None:
    tokenizer_path = tmp_path / "tokenizer"
    _tokenizer(tokenizer_path)
    source = tmp_path / "clean.jsonl"
    output = tmp_path / "packed.jsonl"
    records = [{"text": "short one", "source": "a"}, {"text": "short two", "source": "b"}, {"text": "x" * 100, "source": "c"}]
    source.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    subprocess.run([
        sys.executable, "scripts/pack_jsonl_corpus.py", str(source),
        "--output", str(output), "--tokenizer", str(tokenizer_path),
        "--sequence-length", "32",
    ], check=True)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows
    assert all(row["prepacked"] is True for row in rows)
    assert all(row["token_count"] <= 32 for row in rows)
    tokenizer = Tokenizer.load(tokenizer_path)
    actual = [token for row in rows for token in row["token_ids"]]
    expected = [token for record in records for token in
                tokenizer.encode(record["text"], add_eos=True)]
    assert actual == expected
    assert all(row["token_count"] == 32 for row in rows[:-1])
    assert output.with_suffix(".packing.json").is_file()


@pytest.mark.parametrize("sequence_length", [2, 8, 32])
def test_packed_unicode_is_lossless_through_both_loaders(tmp_path, sequence_length):
    tokenizer_path = tmp_path / "tokenizer"
    _tokenizer(tokenizer_path)
    tokenizer = Tokenizer.load(tokenizer_path)
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    text = "Hello বাংলা हिन्दी 🙂" * 4
    source.write_text(json.dumps({"text": text}) + "\n")
    subprocess.run([sys.executable, "scripts/pack_jsonl_corpus.py", str(source),
                    "--output", str(output), "--tokenizer", str(tokenizer_path),
                    "--sequence-length", str(sequence_length)], check=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    expected = tokenizer.encode(text, add_eos=True)
    for dataset in (TextDataset(rows, tokenizer, max_length=sequence_length),
                    LazyJSONLDataset(output, tokenizer, max_length=sequence_length)):
        actual = [token for index in range(len(dataset)) for token in
                  dataset[index]["input_ids"].tolist()[1:]]
        assert actual == expected
    row = dict(rows[0], tokenizer_fingerprint="wrong")
    with pytest.raises(ValueError, match="fingerprint"):
        TextDataset([row], tokenizer, max_length=sequence_length)
    with pytest.raises(ValueError, match="exceeds max_length"):
        TextDataset([dict(rows[0], token_ids=[10] * sequence_length)],
                    tokenizer, max_length=sequence_length)
