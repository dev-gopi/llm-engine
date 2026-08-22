import json

import torch

from datasets.collator import Collator
from datasets.loader import TextDataset, iter_records
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
    assert torch.equal(batch["attention_mask"], batch["loss_mask"])
    assert torch.all(batch["labels"][~batch["attention_mask"]] == -100)
