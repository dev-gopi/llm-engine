import struct
from pathlib import Path

import torch

from inference.quantized_formats import export_gguf
from schema.grammar import GrammarConstraint, GrammarSpec
from serving.schemas import GenerateRequest


def test_real_gguf_v3_writer(tmp_path: Path):
    out = tmp_path / "model.gguf"
    export_gguf({"linear.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}, out)
    raw = out.read_bytes()
    magic, version, tensors, kv = struct.unpack_from("<4sIQQ", raw, 0)
    assert magic == b"GGUF"
    assert version == 3
    assert tensors == 1
    assert kv >= 2
    assert out.stat().st_size > 64


def test_grammar_constraint():
    grammar = GrammarConstraint(
        GrammarSpec(
            """start: "hello" "/" WORD\n%import common.WORD\n%import common.WS\n%ignore WS"""
        )
    )
    assert grammar.accepts("hello/world")
    assert not grammar.accepts("goodbye/world")


def test_generate_request_cache_context_is_backward_compatible():
    request = GenerateRequest(
        prompt="hello", temperature=0, tenant_id="tenant-a", route="/v1/chat"
    )
    assert request.tenant_id == "tenant-a"
    assert request.route == "/v1/chat"


def test_llama_cpp_minigpt_mapping_and_gguf_export(tmp_path: Path):
    from inference.quantized_formats import (
        export_llama_cpp_gguf,
        minigpt_llama_state_dict,
    )
    from model.gpt import MiniGPT
    from tokenizer.bpe import BYTE_ENCODER
    from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer

    vocab = {
        token: index
        for index, token in enumerate(
            list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
        )
    }
    tokenizer = Tokenizer(
        vocab,
        special_tokens={token: vocab[token] for token in DEFAULT_SPECIAL_TOKENS},
    )
    config = {
        "vocab_size": len(vocab),
        "hidden_size": 16,
        "layers": 2,
        "heads": 4,
        "kv_heads": 2,
        "max_position": 32,
        "position_type": "rotary",
        "rope_base": 10000.0,
        "rope_scaling_type": "none",
        "attention_bias": False,
        "causal_attention": True,
        "qk_norm": False,
        "ffn_hidden_size": 32,
        "ffn_activation": "swiglu",
        "ffn_bias": False,
        "ffn_type": "dense",
        "norm_type": "rms_norm",
        "norm_eps": 1e-5,
        "norm_bias": False,
        "pre_norm": True,
        "lm_head_bias": False,
    }
    model = MiniGPT.from_config(config, device="cpu")
    mapped = minigpt_llama_state_dict(model, config)
    assert "token_embd.weight" in mapped
    assert "blk.0.attn_q.weight" in mapped
    assert "blk.0.ffn_gate.weight" in mapped
    assert mapped["blk.0.ffn_gate.weight"].shape == (32, 16)

    out = export_llama_cpp_gguf(model, config, tokenizer, tmp_path / "model.gguf")
    magic, version, tensors, kv = struct.unpack_from("<4sIQQ", out.read_bytes(), 0)
    assert magic == b"GGUF"
    assert version == 3
    assert tensors == len(mapped)
    assert kv > 10
