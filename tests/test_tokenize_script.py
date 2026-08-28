import json

from scripts.tokenize import iter_corpus


def _write_jsonl(path, texts):
    path.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in texts),
        encoding="utf-8",
    )


def test_iter_corpus_round_robin_reads_every_source(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    third = tmp_path / "third.jsonl"
    _write_jsonl(first, ["a1", "a2", "a3"])
    _write_jsonl(second, ["b1"])
    _write_jsonl(third, ["c1", "c2"])

    texts = list(
        iter_corpus(
            [str(first), str(second), str(third)],
            sampling="round_robin",
        )
    )

    assert texts == ["a1", "b1", "c1", "a2", "c2", "a3"]


def test_iter_corpus_remains_sequential_by_default(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, ["a1", "a2"])
    _write_jsonl(second, ["b1"])

    assert list(iter_corpus([str(first), str(second)])) == ["a1", "a2", "b1"]


def test_iter_corpus_balances_contributed_bytes(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, ["a" * 10, "a2"])
    _write_jsonl(second, ["b", "b2", "b3"])

    texts = list(
        iter_corpus(
            [str(first), str(second)],
            sampling="balanced_bytes",
        )
    )

    # After the first source contributes a large document, the smaller second
    # source catches up before the first source is read again.
    assert texts == ["a" * 10, "b", "b2", "b3", "a2"]
