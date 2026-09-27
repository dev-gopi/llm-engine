import asyncio
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from inference.generator import Generator
from serving import backend as serving_backend
from serving.backend import VLLMServingBackend
from serving.schemas import GenerateRequest, OpenAIChatCompletionRequest
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def make_tokenizer():
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    return Tokenizer(
        vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS}
    )


class BeamToy(nn.Module):
    max_positions = 16

    def __init__(self, vocab_size, b_id, eos_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.b_id = b_id
        self.eos_id = eos_id
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, ids, **kwargs):
        logits = torch.full((*ids.shape, self.vocab_size), -100.0, device=ids.device)
        last = int(ids[0, -1].item())
        nxt = self.eos_id if last == self.b_id else self.b_id
        logits[:, -1, nxt] = 100.0
        return logits


def test_native_generator_beam_search_is_exposed():
    tok = make_tokenizer()
    b = tok.token_to_id(BYTE_ENCODER[ord("b")])
    eos = tok.token_to_id("<|eos|>")
    result = Generator(
        BeamToy(tok.vocab_size, b, eos), tok, device="cpu"
    ).generate_beam("a", max_tokens=4, num_beams=2)
    assert result.text == "b"
    assert result.finish_reason == "stop"
    assert result.token_ids == (b,)


def test_decoding_strategy_contracts():
    req = GenerateRequest(
        prompt="x", decoding_strategy="beam", num_beams=3, length_penalty=0.8
    )
    assert req.num_beams == 3
    with pytest.raises(ValueError, match="draft_model_id"):
        GenerateRequest(prompt="x", decoding_strategy="speculative")
    with pytest.raises(ValueError, match="logprobs"):
        GenerateRequest(prompt="x", decoding_strategy="beam", logprobs=True)
    chat = OpenAIChatCompletionRequest(
        model="gopi",
        messages=[{"role": "user", "content": "x"}],
        decoding_strategy="beam",
    )
    internal = chat.generation_request("gopi")
    assert internal.decoding_strategy == "beam"
    assert internal.num_beams == 4


def test_vllm_serving_backend_maps_output(monkeypatch):
    class FakeAdapter:
        def __init__(self, model, **options):
            self.model = model
            self.options = options

        def generate(self, prompts, **kwargs):
            assert prompts == ["hello"]
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[
                        SimpleNamespace(
                            text="world", token_ids=[3, 4], finish_reason="stop"
                        )
                    ],
                )
            ]

    monkeypatch.setattr(serving_backend, "NativeVLLMBackend", FakeAdapter)
    backend = VLLMServingBackend("demo")
    result = asyncio.run(
        backend.generate(GenerateRequest(prompt="hello", temperature=0))
    )
    assert result.text == "world"
    assert result.prompt_tokens == 2
    assert result.completion_tokens == 2


def test_reload_candidate_accepts_vllm(monkeypatch):
    class FakeServing:
        ready = True

        def __init__(self, model, **options):
            self.model = model
            self.options = options

    monkeypatch.setattr(serving_backend, "VLLMServingBackend", FakeServing)
    monkeypatch.setenv("GOPI_BACKEND", "vllm")
    monkeypatch.setenv("GOPI_VLLM_MODEL", "demo-model")
    monkeypatch.setenv("GOPI_VLLM_TENSOR_PARALLEL_SIZE", "2")
    monkeypatch.setenv("GOPI_VLLM_PIPELINE_PARALLEL_SIZE", "3")
    monkeypatch.setenv("GOPI_VLLM_ENABLE_EXPERT_PARALLEL", "true")
    backend, version = serving_backend._reload_candidate()
    assert backend.model == "demo-model"
    assert backend.options["tensor_parallel_size"] == 2
    assert backend.options["pipeline_parallel_size"] == 3
    assert backend.options["enable_expert_parallel"] is True
    assert version == "vllm:demo-model"


def test_vllm_serving_backend_uses_chat_api_when_messages_present(monkeypatch):
    class FakeAdapter:
        def __init__(self, model, **options):
            pass

        def chat(self, conversations, **kwargs):
            assert conversations == [[{"role": "user", "content": "hello"}]]
            return [
                SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[
                        SimpleNamespace(text="hi", token_ids=[2], finish_reason="stop")
                    ],
                )
            ]

        def generate(self, prompts, **kwargs):
            raise AssertionError("chat requests should use vLLM chat")

    monkeypatch.setattr(serving_backend, "NativeVLLMBackend", FakeAdapter)
    backend = VLLMServingBackend("demo")
    request = GenerateRequest(prompt="hello")
    request._chat_messages = [{"role": "user", "content": "hello"}]
    result = asyncio.run(backend.generate(request))
    assert result.text == "hi"
