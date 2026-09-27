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
