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
from model.kv_cache import StaticLayerKVCache
from training.checkpoint import save_checkpoint


def make_tokenizer() -> Tokenizer:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    specials = {piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS}
    return Tokenizer(vocab, special_tokens=specials)


def test_static_kv_cache_appends_without_reallocating_storage() -> None:
    initial_key = torch.randn(1, 2, 3, 4)
    initial_value = torch.randn(1, 2, 3, 4)
    cache = StaticLayerKVCache(initial_key, initial_value, capacity=8)
    key_pointer = cache.key.data_ptr()
    value_pointer = cache.value.data_ptr()
    next_key = torch.randn(1, 2, 1, 4)
    next_value = torch.randn(1, 2, 1, 4)

    keys, values = cache.append(next_key, next_value)

    assert cache.key.data_ptr() == key_pointer
    assert cache.value.data_ptr() == value_pointer
    assert cache.length == 4
    torch.testing.assert_close(keys[:, :, :3], initial_key)
    torch.testing.assert_close(values[:, :, 3:], next_value)


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


class PredictSpecialThenB(nn.Module):
    max_positions = 8

    def __init__(self, vocab_size: int, special_id: int, b_id: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.special_id = special_id
        self.b_id = b_id
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, token_ids: torch.Tensor, *, past_key_values=None, use_cache=False):
        logits = torch.full((*token_ids.shape, self.vocab_size), -100.0, device=token_ids.device)
        logits[:, -1, self.special_id] = 100.0
        logits[:, -1, self.b_id] = 90.0
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


def test_generator_suppresses_non_eos_special_tokens() -> None:
    tokenizer = make_tokenizer()
    b_id = tokenizer.token_to_id(BYTE_ENCODER[ord("b")])
    model = PredictSpecialThenB(
        tokenizer.vocab_size,
        tokenizer.token_to_id("<|user|>"),
        b_id,
    )

    result = Generator(model, tokenizer, device="cpu").generate(
        "a", max_tokens=1, temperature=0,
    )

    assert result.text == "b"
    assert result.token_ids == (b_id,)


def test_special_token_suppression_applies_to_every_decode_mode() -> None:
    tokenizer = make_tokenizer()
    b_id = tokenizer.token_to_id(BYTE_ENCODER[ord("b")])
    generator = Generator(PredictSpecialThenB(
        tokenizer.vocab_size,
        tokenizer.token_to_id("<|assistant|>"),
        b_id,
    ), tokenizer, device="cpu")

    batch = generator.generate_batch(["a"], max_tokens=1, temperature=0)
    stream = list(generator.stream("a", max_tokens=1, temperature=0))
    state = generator.start_batched_stream("a", max_tokens=1, temperature=0)
    streamed_step, done = generator.decode_batched_stream([state])[0]

    assert batch[0].text == "b"
    assert "".join(step.token for step in stream) == "b"
    assert streamed_step.token == "b"
    assert done


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
    with pytest.raises(ValueError, match="no_repeat_ngram_size"):
        generator.generate("a", no_repeat_ngram_size=-1)
    with pytest.raises(ValueError, match="min_tokens"):
        list(generator.stream("a", min_tokens=-1))


def test_no_repeat_ngram_bans_only_tokens_that_complete_a_duplicate() -> None:
    logits = torch.zeros((1, 8))

    Generator._apply_no_repeat_ngram(logits, [1, 2, 3, 1, 2], 3)

    assert torch.isneginf(logits[0, 3])
    assert torch.isfinite(logits[0, 2])


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


def test_minigpt_generation_projects_only_last_token_and_skips_final_forward():
    tokenizer = make_tokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=16, layers=1,
                    heads=2, max_pos=64)
    generator = Generator(model, tokenizer, device="cpu")
    shapes = []
    hook = model.head.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
    try:
        result = generator.generate("hello", max_tokens=1, temperature=0)
        assert len(result.token_ids) == 1
        assert len(shapes) == 1
        assert shapes[0][1] == 1
        shapes.clear()
        list(generator.stream("hello", max_tokens=1, temperature=0))
        assert len(shapes) == 1
    finally:
        hook.remove()


@pytest.mark.parametrize("streaming", [False, True])
def test_generation_does_not_mutate_prefix_logits(streaming):
    tokenizer = make_tokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=16, layers=1,
                    heads=2, max_pos=64)
    generator = Generator(model, tokenizer, device="cpu", prefix_cache_capacity=2)
    prompt_ids = tokenizer.encode("hello", add_bos=True)
    with torch.inference_mode():
        generator._prefill(prompt_ids)
        cached_logits = generator.prefix_cache.get(tuple(prompt_ids))[0]
        original = cached_logits.clone()
    options = dict(max_tokens=2, temperature=0, repetition_penalty=1.5)
    if streaming:
        first = list(generator.stream("hello", **options))
        second = list(generator.stream("hello", **options))
    else:
        first = generator.generate("hello", **options)
        second = generator.generate("hello", **options)
    assert first == second
    torch.testing.assert_close(cached_logits, original)


@pytest.mark.parametrize("temperature", [0, 1])
@pytest.mark.parametrize("values", [[float("-inf"), float("-inf")], [float("nan"), 0.0], [float("inf"), 0.0]])
def test_sampler_rejects_invalid_rows_for_greedy_and_sampling(temperature, values):
    with pytest.raises(ValueError, match="logits"):
        TopKSampler()(torch.tensor([values]), temperature=temperature)


@pytest.mark.parametrize("temperature", [0, 1])
def test_sampler_preserves_blocked_tokens(temperature):
    result = TopKSampler()(torch.tensor([[float("-inf"), 1.0]]), temperature=temperature)
    assert result.tolist() == [1]


@pytest.mark.parametrize("top_k", [1.5, True, -1])
def test_sampler_rejects_noninteger_or_negative_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        TopKSampler()(torch.zeros(1, 4), top_k=top_k)


@pytest.mark.parametrize("position_type", ["learned", "rotary", "sinusoidal"])
def test_batch_static_cache_reuses_storage_and_matches_single_generation(position_type, monkeypatch):
    torch.manual_seed(7)
    tokenizer = make_tokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=16, layers=2, heads=2,
                    max_pos=32, position_type=position_type)
    generator = Generator(model, tokenizer, device="cpu")
    options = dict(max_tokens=5, min_tokens=5, temperature=0,
                   repetition_penalty=1.0, no_repeat_ngram_size=0)
    expected = [generator.generate(prompt, **options) for prompt in ("a", "b")]
    pointers = []
    forward = model.forward

    def recording_forward(*args, **kwargs):
        cache = kwargs.get("past_key_values")
        if cache:
            assert all(isinstance(layer, StaticLayerKVCache) for layer in cache)
            pointers.append(tuple(layer.key.data_ptr() for layer in cache))
        return forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", recording_forward)
    decode = tokenizer.decode
    decoded_lengths = []

    def recording_decode(ids, **kwargs):
        decoded_lengths.append(len(ids))
        return decode(ids, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", recording_decode)
    actual = generator.generate_batch(["a", "b"], **options)
    assert actual == expected
    assert len(pointers) == 4
    assert all(value == pointers[0] for value in pointers)
    assert decoded_lengths == [5, 5]


def test_batch_static_cache_compacts_finished_rows_and_preserves_stop_text(monkeypatch):
    tokenizer = make_tokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=8, layers=1, heads=2, max_pos=16)
    generator = Generator(model, tokenizer, device="cpu")
    # First request reaches its stop string immediately; second continues.
    a = tokenizer.encode("a")[0]
    b = tokenizer.encode("b")[0]
    c = tokenizer.encode("c")[0]
    samples = iter([a, b, c, c])
    monkeypatch.setattr(generator, "sampler", lambda *args, **kwargs: torch.tensor([next(samples)]))
    shapes = []
    forward = model.forward

    def recording_forward(ids, **kwargs):
        shapes.append(ids.shape[0])
        return forward(ids, **kwargs)

    monkeypatch.setattr(model, "forward", recording_forward)
    results = generator.generate_batch(["x", "y"], max_tokens=3, stop=["a"], temperature=0)
    assert [result.text for result in results] == ["", "bcc"]
    assert [result.finish_reason for result in results] == ["stop", "length"]
    assert shapes == [2, 1, 1]


@pytest.mark.parametrize("completion,expected,reason", [
    ("bEND", "b", "stop"),
    ("bENx", "bENx", "length"),
])
def test_batched_stream_holds_stop_prefix_and_flushes_at_limit(monkeypatch, completion, expected, reason):
    tokenizer = make_tokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=8, layers=1, heads=2, max_pos=16)
    generator = Generator(model, tokenizer, device="cpu")
    tokens = iter(tokenizer.encode(completion))
    monkeypatch.setattr(generator, "sampler", lambda *args, **kwargs: torch.tensor([next(tokens)]))
    state = generator.start_batched_stream("a", max_tokens=4, stop=["END"])
    events = []
    for _ in range(4):
        event, done = generator.decode_batched_stream([state])[0]
        events.append(event)
        if done:
            break
    assert "".join(event.token for event in events) == expected
    assert events[-1].finish_reason == reason
    generator.release_batched_stream(state)


def test_min_p_filters_relative_to_best_probability(monkeypatch):
    captured = []

    def capture(probabilities, *args, **kwargs):
        captured.append(probabilities)
        return probabilities.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", capture)
    logits = torch.tensor([[0.6, 0.3, 0.1]]).log()
    TopKSampler()(logits, min_p=0.4)
    torch.testing.assert_close(captured[0], torch.tensor([[2 / 3, 1 / 3, 0.0]]))
    TopKSampler()(logits, min_p=1.0, top_p=0.9, top_k=2)
    torch.testing.assert_close(captured[1], torch.tensor([[1.0, 0.0, 0.0]]))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_min_p_rejects_invalid_cutoffs(value):
    with pytest.raises(ValueError, match="min_p"):
        TopKSampler()(torch.zeros(1, 3), min_p=value, temperature=0)


def test_min_p_reaches_all_generation_modes(monkeypatch):
    tokenizer = make_tokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=8, layers=1, heads=2, max_pos=16)
    generator = Generator(model, tokenizer, device="cpu")
    observed = []

    def sample(logits, **kwargs):
        observed.append(kwargs["min_p"])
        return torch.tensor([tokenizer.encode("b")[0]])

    monkeypatch.setattr(generator, "sampler", sample)
    options = dict(max_tokens=1, min_p=0.15)
    generator.generate("a", **options)
    generator.generate_batch(["a"], **options)
    list(generator.stream("a", **options))
    state = generator.start_batched_stream("a", **options)
    generator.decode_batched_stream([state])
    generator.release_batched_stream(state)
    assert observed == [0.15] * 4


def test_min_p_api_validation_and_chat_conversion():
    from serving.schemas import OpenAIChatCompletionRequest
    from pydantic import ValidationError
    request = OpenAIChatCompletionRequest(
        model="gopi", messages=[{"role": "user", "content": "hello"}], min_p=0.2,
    )
    assert request.generation_request("gopi").min_p == 0.2
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hello", min_p=1.1)
