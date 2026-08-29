import json

import pytest
import torch

from datasets.collator import Collator
from datasets.loader import LazyJSONLDataset, TextDataset, iter_records
from datasets.preprocessor import clean, format_messages
from datasets.sampler import Sampler
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def tokenizer() -> Tokenizer:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    return Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})


def test_preprocessor_formats_chat_and_normalizes_text() -> None:
    assert clean(" A\r\n\r\n\r\nB\x00 ") == "A\n\nB"
    rendered = format_messages([{"role": "user", "content": "Hello"}], add_generation_prompt=True)
    assert rendered == "<|user|>\nHello\n<|assistant|>\n"


def test_dataset_reader_tokenization_collator_and_sampler(tmp_path) -> None:
    source = tmp_path / "records.jsonl"
    records = [
        {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]},
        {"text": "a longer plain text sample"},
    ]
    source.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    dataset = TextDataset(iter_records(source), tokenizer(), max_length=64)
    assert len(dataset) == 2
    lengths = [dataset[index]["input_ids"].numel() for index in range(len(dataset))]
    batches = list(Sampler(lengths, batch_size=2, seed=3))
    batch = Collator(tokenizer().token_to_id("<|pad|>"), pad_to_multiple_of=8)(
        [dataset[index] for index in batches[0]]
    )
    assert batch["input_ids"].shape[1] % 8 == 0
    assert torch.all(batch["loss_mask"] <= batch["attention_mask"])
    assert torch.any(batch["attention_mask"] & ~batch["loss_mask"])
    assert torch.all(batch["labels"][~batch["attention_mask"]] == -100)


def test_sampler_shards_ranks_without_overlap() -> None:
    lengths = list(range(12))
    first = {index for batch in Sampler(lengths, 2, shuffle=False, rank=0, world_size=2) for index in batch}
    second = {index for batch in Sampler(lengths, 2, shuffle=False, rank=1, world_size=2) for index in batch}
    assert first.isdisjoint(second)
    assert first | second == set(range(12))


def test_sampler_resume_skips_completed_batches() -> None:
    sampler = Sampler(list(range(8)), 2, shuffle=False)
    sampler.load_state_dict({"epoch": 3, "start_batch": 2})
    assert list(sampler) == [[4, 5], [6, 7]]


def test_weighted_sampler_uses_bounded_deterministic_epoch() -> None:
    sampler = Sampler(
        [1, 1, 1, 1], 2, seed=7,
        sampling_weights=[0.0, 0.0, 1.0, 1.0], num_samples=10,
    )
    first = list(sampler)
    assert len(first) == 5
    assert {index for batch in first for index in batch} <= {2, 3}
    assert first == list(Sampler(
        [1, 1, 1, 1], 2, seed=7,
        sampling_weights=[0.0, 0.0, 1.0, 1.0], num_samples=10,
    ))


def test_lazy_jsonl_indexes_without_eager_tokenization(tmp_path) -> None:
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    dataset = LazyJSONLDataset(source, tokenizer(), max_length=16)
    assert len(dataset) == 1
    assert dataset[0]["input_ids"].numel() >= 2


def test_text_dataset_rejects_unusable_records_instead_of_skipping() -> None:
    with pytest.raises(ValueError, match="invalid dataset record at index 0"):
        TextDataset([{"unsupported": "value"}], tokenizer(), max_length=16)


def test_truncated_user_turn_does_not_inject_synthetic_eos() -> None:
    dataset = TextDataset(
        [{"messages": [{"role": "user", "content": "a" * 100}]}],
        tokenizer(),
        max_length=16,
    )

    example = dataset[0]
    assert example["input_ids"][-1].item() != tokenizer().token_to_id("<|eos|>")
    assert not example["loss_mask"][-1].item()


def test_truncated_assistant_turn_keeps_content_instead_of_eos() -> None:
    dataset = TextDataset(
        [{"messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "a" * 100},
        ]}],
        tokenizer(),
        max_length=32,
    )

    example = dataset[0]
    assert example["input_ids"][-1].item() != tokenizer().token_to_id("<|eos|>")
    assert example["loss_mask"][-1].item()


def test_complete_example_retains_real_eos() -> None:
    dataset = TextDataset(
        [{"messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]}],
        tokenizer(),
        max_length=128,
    )

    example = dataset[0]
    assert example["input_ids"][-1].item() == tokenizer().token_to_id("<|eos|>")
    assert example["loss_mask"][-1].item()


def test_lazy_jsonl_reports_malformed_record_location_without_substitution(tmp_path) -> None:
    source = tmp_path / "data.jsonl"
    source.write_text('{"text": "valid"}\nnot-json\n', encoding="utf-8")
    dataset = LazyJSONLDataset(source, tokenizer(), max_length=16)

    with pytest.raises(ValueError, match=r"invalid JSON at .*data\.jsonl:2"):
        dataset[1]


def test_lazy_jsonl_reports_unusable_record_location_without_substitution(tmp_path) -> None:
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps({"unsupported": "value"}) + "\n", encoding="utf-8")
    dataset = LazyJSONLDataset(source, tokenizer(), max_length=16)

    with pytest.raises(ValueError, match=r"unusable dataset record at .*data\.jsonl:1"):
        dataset[0]
