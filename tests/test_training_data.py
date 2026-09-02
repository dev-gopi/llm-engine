import json

import numpy as np
import pytest

from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer
from training.data import _mixture_groups, _mixture_name, build_loader


def test_dataset_mixture_weights_are_dataset_level_probabilities() -> None:
    groups = _mixture_groups(
        ["data/small/train.jsonl", "data/large/train.jsonl"],
        [2, 8],
        {"dataset_weights": {"small": 0.25, "large": 0.75}},
    )
    assert groups == [(0, 2, 0.25), (2, 10, 0.75)]


def test_mixture_name_distinguishes_files_in_shared_directory() -> None:
    configured = {"wikipedia_en": 0.01, "wikipedia_bn": 0.02}

    assert _mixture_name("data/rag/wikipedia/wikipedia-en.jsonl", configured) == "wikipedia_en"
    assert _mixture_name("data/rag/wikipedia/wikipedia-bn.jsonl", configured) == "wikipedia_bn"


def test_loader_can_pad_batches_for_tensor_core_shapes(tmp_path) -> None:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    tokenizer = Tokenizer(
        vocab,
        special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS},
    )
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({"text": "five"}) + "\n", encoding="utf-8")

    loader = build_loader(
        [source], tokenizer,
        {"batch_size": 1, "max_sequence_length": 32, "pad_to_multiple_of": 8},
        shuffle=False,
    )

    assert next(iter(loader))["input_ids"].shape[1] % 8 == 0

    with pytest.raises(ValueError, match="max_sequence_length must be divisible"):
        build_loader(
            [source], tokenizer,
            {"batch_size": 1, "max_sequence_length": 30, "pad_to_multiple_of": 8},
            shuffle=False,
        )


def test_token_shards_reject_different_same_size_tokenizer(tmp_path) -> None:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    tokenizer = Tokenizer(
        vocab,
        special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS},
    )
    np.arange(8, dtype=np.uint32).tofile(tmp_path / "tokens.bin")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "gopi-token-shards-v1",
        "dtype": "uint32",
        "sequence_length": 8,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": "different-tokenizer",
        "shards": [{"file": "tokens.bin", "sequences": 1}],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        build_loader(
            [tmp_path / "manifest.json"], tokenizer,
            {"batch_size": 1, "max_sequence_length": 8}, shuffle=False,
        )


def test_multiple_token_shards_preserve_dataset_mixture_groups(tmp_path) -> None:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    tokenizer = Tokenizer(
        vocab,
        special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS},
    )
    manifests = []
    for name, count in (("tinystories", 2), ("wikitext_103", 3)):
        directory = tmp_path / name
        directory.mkdir()
        np.arange(count * 8, dtype=np.uint16).tofile(directory / "tokens.bin")
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({
            "format": "gopi-token-shards-v1",
            "dtype": "uint16",
            "sequence_length": 8,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "shards": [{"file": "tokens.bin", "sequences": count}],
        }), encoding="utf-8")
        manifests.append(manifest)

    loader = build_loader(
        manifests,
        tokenizer,
        {
            "batch_size": 1,
            "max_sequence_length": 8,
            "dataset_weights": {"tinystories": 0.35, "wikitext_103": 0.65},
            "samples_per_epoch": 10,
            "num_workers": 1,
            "persistent_workers": True,
            "prefetch_factor": 3,
        },
        shuffle=True,
    )

    assert loader.dataset.dataset_sizes == [2, 3]
    assert loader.batch_sampler.sampling_groups == [
        (0, 2, 0.35), (2, 5, 0.65),
    ]
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3
