import asyncio

import torch
from torch import nn

from inference.generator import Generator
from inference.sampler import TopKSampler
from model.gpt import MiniGPT
from serving.backend import ConfiguredModelBackend
from serving.schemas import GenerateRequest
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer
from training.checkpoint import save_checkpoint


def make_tokenizer() -> Tokenizer:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    specials = {piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS}
    return Tokenizer(vocab, special_tokens=specials)


class PredictBThenEos(nn.Module):
    max_positions = 8

    def __init__(self, vocab_size: int, b_id: int, eos_id: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.b_id = b_id
        self.eos_id = eos_id
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, token_ids: torch.Tensor, *, past_key_values=None, use_cache=False):
        logits = torch.full((*token_ids.shape, self.vocab_size), -100.0, device=token_ids.device)
        next_id = self.b_id if past_key_values is None else self.eos_id
        logits[:, -1, next_id] = 100.0
        if use_cache:
            length = token_ids.shape[1] + (past_key_values[0][0].shape[2] if past_key_values else 0)
            cache = torch.zeros((1, 1, length, 1), device=token_ids.device)
            return logits, ((cache, cache.clone()),)
        return logits


def test_generator_connects_tokenizer_model_sampler_and_decoder() -> None:
    tokenizer = make_tokenizer()
    model = PredictBThenEos(
        tokenizer.vocab_size,
        tokenizer.token_to_id(BYTE_ENCODER[ord("b")]),
        tokenizer.token_to_id("<|eos|>"),
    )
    result = Generator(model, tokenizer, device="cpu").generate(
        "a", max_tokens=4, temperature=0
    )
    assert result.text == "b"
    assert result.prompt_tokens == 2
    assert result.finish_reason == "stop"
    assert len(result.token_ids) == 1


def test_generator_stream_yields_before_final_event() -> None:
    tokenizer = make_tokenizer()
    model = PredictBThenEos(tokenizer.vocab_size, tokenizer.token_to_id(BYTE_ENCODER[ord("b")]), tokenizer.token_to_id("<|eos|>"))
    events = list(Generator(model, tokenizer, device="cpu").stream("a", max_tokens=4, temperature=0))
    assert events[0].token == "b"
    assert events[0].finish_reason is None
    assert events[-1].finish_reason == "stop"


def test_sampler_supports_greedy_and_seeded_sampling() -> None:
    sampler = TopKSampler()
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    assert sampler(logits, temperature=0).item() == 1
    first = sampler(logits, temperature=1, generator=torch.Generator().manual_seed(5))
    second = sampler(logits, temperature=1, generator=torch.Generator().manual_seed(5))
    assert torch.equal(first, second)


def test_checkpoint_to_serving_backend_integration(tmp_path) -> None:
    tokenizer = make_tokenizer()
    tokenizer.save(tmp_path / "tokenizer")
    config = {
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": 8,
        "layers": 1,
        "heads": 2,
        "max_position": 8,
        "position_type": "learned",
        "ffn_hidden_size": 16,
        "ffn_multiple_of": 1,
    }
    config_path = tmp_path / "model.yaml"
    config_path.write_text("\n".join(f"{key}: {value}" for key, value in config.items()), encoding="utf-8")
    model = MiniGPT.from_config(config)
    checkpoint = save_checkpoint(tmp_path / "model.pt", model, step=7)
    backend = ConfiguredModelBackend(
        model_config=config_path,
        tokenizer_path=tmp_path / "tokenizer",
        checkpoint_path=checkpoint,
        device="cpu",
    )

    async def exercise() -> None:
        await backend.startup()
        assert backend.ready
        response = await backend.generate(
            GenerateRequest(prompt="hello", max_tokens=1, temperature=0)
        )
        assert response.prompt_tokens > 0
        assert response.completion_tokens in (0, 1)
        await backend.shutdown()
        assert not backend.ready

    asyncio.run(exercise())
