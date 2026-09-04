import json
import subprocess
import sys
from pathlib import Path

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
    assert all(row["text"].endswith("<|eos|>") for row in rows)
    assert output.with_suffix(".packing.json").is_file()
