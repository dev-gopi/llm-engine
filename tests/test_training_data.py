import json
from pathlib import Path

import numpy as np
import pytest

from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer
from training.data import _mixture_groups, _mixture_name, build_loader, interleave_loaders
from datasets.sampler import CurriculumSchedule, CurriculumStage, Sampler


def test_validation_sources_are_interleaved_before_batch_cap() -> None:
    loader = interleave_loaders([["a1", "a2", "a3"], ["b1", "b2"]])

    assert list(loader) == ["a1", "b1", "a2", "b2", "a3"]


def test_dataset_mixture_weights_are_dataset_level_probabilities() -> None:
    groups = _mixture_groups(
        ["data/small/train.jsonl", "data/large/train.jsonl"],
        [2, 8],
        {"dataset_weights": {"small": 0.25, "large": 0.75}},
    )
    assert groups == [(0, 2, 0.25), (2, 10, 0.75)]


@pytest.mark.parametrize("weights", [
    {"recovery_chat": 0.5, "aya_hindi": 0.5},
    {"chat": 0.5},
    {"chat": 0.5, "aya_hindi": 0.5, "typo": 0.1},
])
def test_mixture_rejects_silent_fallback_weights(weights):
    with pytest.raises(ValueError, match="dataset_weights must match"):
        _mixture_groups(["recovery_sft/chat/train.jsonl", "aya_hindi/train.jsonl"],
                        [10, 20], {"dataset_weights": weights})


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1])
def test_mixture_rejects_nonfinite_or_negative_weights(weight):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _mixture_groups(["chat/train.jsonl"], [10], {"dataset_weights": {"chat": weight}})


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


def test_validation_worker_budget_is_independent(tmp_path):
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    tokenizer = Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({"text": "hello"}) + "\n")
    config = dict(batch_size=1, max_sequence_length=32, num_workers=4,
                  persistent_workers=True, validation_num_workers=1,
                  validation_persistent_workers=False)
    train = build_loader([source], tokenizer, config, shuffle=True)
    validation = build_loader([source], tokenizer, config, shuffle=False)
    assert train.num_workers == 4 and train.persistent_workers
    assert validation.num_workers == 1 and not validation.persistent_workers
    config["validation_num_workers"] = 0
    assert build_loader([source], tokenizer, config, shuffle=False).num_workers == 0


def test_curriculum_schedule_updates_and_restores_sampler_group_weights() -> None:
    sampler = Sampler(
        [3, 4, 5, 6], batch_size=1, sampling_groups=[(0, 2, 0.1), (2, 4, 0.9)],
    )
    schedule = CurriculumSchedule((
        CurriculumStage(0, (0.1, 0.9)), CurriculumStage(2, (0.4, 0.6)),
    ))
    stage_index, stage = schedule.stage_for_epoch(2)
    sampler.set_sampling_group_weights(stage.weights)
    restored = Sampler(
        [3, 4, 5, 6], batch_size=1, sampling_groups=[(0, 2, 0.1), (2, 4, 0.9)],
    )
    restored.load_state_dict(sampler.state_dict())

    assert stage_index == 1
    assert [weight for _, _, weight in restored.sampling_groups] == [0.4, 0.6]


def test_pretraining_curriculum_matches_active_source_count() -> None:
    root = Path(__file__).resolve().parents[1]
    import yaml
    config = yaml.safe_load((root / "configs/pretraining.gpu.yaml").read_text(encoding="utf-8"))
    schedule = CurriculumSchedule.from_config(config["curriculum"])
    assert len(schedule.stages[0].weights) == len(config["train_files"])
