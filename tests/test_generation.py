import asyncio

import torch
import pytest
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


def test_all_generation_modes_validate_unsafe_options() -> None:
    tokenizer = make_tokenizer()
    model = PredictBThenEos(
        tokenizer.vocab_size, tokenizer.token_to_id(BYTE_ENCODER[ord("b")]),
        tokenizer.token_to_id("<|eos|>"),
    )
    generator = Generator(model, tokenizer, device="cpu")
    with pytest.raises(ValueError, match="max_tokens"):
        list(generator.stream("a", max_tokens=0))
    with pytest.raises(ValueError, match="repetition_penalty"):
        generator.generate_batch(["a"], repetition_penalty=0)


def test_generator_reuses_prefix_cache_without_repeating_prefill() -> None:
    tokenizer = make_tokenizer()
    model = PredictBThenEos(
        tokenizer.vocab_size, tokenizer.token_to_id(BYTE_ENCODER[ord("b")]),
        tokenizer.token_to_id("<|eos|>"),
    )
    generator = Generator(model, tokenizer, device="cpu", prefix_cache_capacity=2)
    generator.generate("a", max_tokens=2, temperature=0)
    generator.generate("a", max_tokens=2, temperature=0)
    assert generator.prefix_cache_misses == 1
    assert generator.prefix_cache_hits == 1


def test_generator_reuses_paged_prefix_cache() -> None:
    tokenizer = make_tokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=8, layers=1, heads=2, max_pos=32)
    generator = Generator(
        model, tokenizer, device="cpu", prefix_cache_capacity=2,
        paged_kv_pages=8, paged_kv_page_size=4,
    )
    first = generator.generate("hello", max_tokens=1, temperature=0)
    second = generator.generate("hello", max_tokens=1, temperature=0)
    assert generator.prefix_cache_hits == 1
    assert second.token_ids == first.token_ids


def test_active_paged_cache_appends_and_reclaims_pages() -> None:
    tokenizer = make_tokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=8, layers=1, heads=2, max_pos=32)
    generator = Generator(
        model, tokenizer, device="cpu", paged_kv_pages=8, paged_kv_page_size=4,
    )
    available = len(generator.paged_kv_allocator.free_pages)
    state = generator.start_batched_stream("hello", max_tokens=2, temperature=0)
    assert state.page_request_id in generator.paged_kv_allocator.tables
    before = generator.paged_kv_allocator.lengths[state.page_request_id]
    generator.decode_batched_stream([state])
    if state.page_request_id is not None:
        assert generator.paged_kv_allocator.lengths[state.page_request_id] >= before
    generator.release_batched_stream(state)
    assert state.page_request_id is None
    assert len(generator.paged_kv_allocator.free_pages) == available


def test_generator_tensor_batches_equal_length_prompts() -> None:
    tokenizer = make_tokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=8, layers=1, heads=2, max_pos=32)
    seen_batches: list[int] = []
    original = model.forward
    def recording_forward(token_ids, *args, **kwargs):
        seen_batches.append(token_ids.shape[0])
        return original(token_ids, *args, **kwargs)
    model.forward = recording_forward
    results = Generator(model, tokenizer, device="cpu").generate_batch(
        ["hello", "world"], max_tokens=2, temperature=0
    )
    assert len(results) == 2
    assert 2 in seen_batches


@pytest.mark.parametrize("position_type", ["learned", "rotary"])
def test_token_step_generation_batches_different_prompt_lengths(position_type) -> None:
    tokenizer = make_tokenizer()
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size, dim=8, layers=1, heads=2,
        max_pos=32, position_type=position_type,
    )
    seen_batches = []
    original = model.forward

    def recording_forward(token_ids, *args, **kwargs):
        seen_batches.append(token_ids.shape[0])
        return original(token_ids, *args, **kwargs)

    model.forward = recording_forward
    generator = Generator(model, tokenizer, device="cpu")
    states = [
        generator.start_batched_stream("a", max_tokens=2, temperature=0),
        generator.start_batched_stream("longer", max_tokens=2, temperature=0),
    ]
    first = generator.decode_batched_stream(states)
    assert len(first) == 2
    assert all(not done for _, done in first)
    second = generator.decode_batched_stream(states)
    assert all(done and step.finish_reason in {"length", "stop"} for step, done in second)
    assert 2 in seen_batches


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
        "max_position": 64,
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
        assert response.completion_tokens >= 0
        await backend.shutdown()
        assert not backend.ready

    asyncio.run(exercise())


def test_configured_backend_token_step_adapter(tmp_path) -> None:
    tokenizer = make_tokenizer()
    tokenizer.save(tmp_path / "tokenizer")
    config = {
        "vocab_size": tokenizer.vocab_size, "hidden_size": 8, "layers": 1,
        "heads": 2, "max_position": 64, "position_type": "rotary",
        "ffn_hidden_size": 16, "ffn_multiple_of": 1,
    }
    config_path = tmp_path / "model.yaml"
    config_path.write_text("\n".join(f"{key}: {value}" for key, value in config.items()))
    checkpoint = save_checkpoint(tmp_path / "model.pt", MiniGPT.from_config(config))
    backend = ConfiguredModelBackend(
        model_config=config_path, tokenizer_path=tmp_path / "tokenizer",
        checkpoint_path=checkpoint, device="cpu",
    )

    async def exercise():
        await backend.startup()
        states = [
            await backend.start_stream(GenerateRequest(prompt=value, max_tokens=2, temperature=0))
            for value in ("a", "longer")
        ]
        events = [[], []]
        active = list(enumerate(states))
        while active:
            output = await backend.decode_stream_batch([state for _, state in active])
            survivors = []
            for (original, state), (event, done) in zip(active, output, strict=True):
                if event is not None:
                    events[original].append(event)
                if done:
                    await backend.release_stream(state)
                else:
                    survivors.append((original, state))
            active = survivors
        await backend.shutdown()
        return events

    events = asyncio.run(exercise())
    assert all(values[-1].finish_reason is not None for values in events)
    assert backend._session_locks == {}
