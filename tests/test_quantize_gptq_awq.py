from __future__ import annotations

import json

import pytest
import torch

from model.gpt import MiniGPT
from scripts.export import export_model
from scripts.quantize_awq import quantize_awq_linear, run_minigpt_awq_conversion
from scripts.quantize_gptq import quantize_gptq_linear, run_minigpt_gptq_conversion


@pytest.fixture
def tiny_minigpt():
    config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "layers": 2,
        "heads": 2,
        "kv_heads": 2,
        "max_position": 64,
    }
    model = MiniGPT.from_config(config, device="cpu")
    return model, config


def test_quantize_gptq_linear():
    weight = torch.randn(32, 64)
    packed, scales, zeros = quantize_gptq_linear(weight, bits=4, group_size=32)
    assert packed.shape[0] == 32
    assert packed.shape[1] == 32  # packed 4-bit into uint8 pairs: 64 // 2
    assert scales.shape[0] == 32
    assert scales.shape[1] == 2  # 64 // 32 groups
    assert zeros.shape == scales.shape


def test_quantize_awq_linear():
    weight = torch.randn(32, 64)
    act_scales = torch.ones(64)
    packed, scales, zeros = quantize_awq_linear(
        weight, bits=4, group_size=32, activation_scales=act_scales
    )
    assert packed.shape[0] == 32
    assert packed.shape[1] == 32
    assert scales.shape[0] == 32
    assert scales.shape[1] == 2
    assert zeros.shape == scales.shape


def test_run_minigpt_gptq_conversion(tiny_minigpt, tmp_path):
    model, config = tiny_minigpt
    out_file = tmp_path / "model.gptq.safetensors"
    art, man = run_minigpt_gptq_conversion(model, out_file, config, bits=4)
    assert art.is_file()
    assert man.is_file()
    manifest_data = json.loads(man.read_text(encoding="utf-8"))
    assert manifest_data["format"] == "gptq"
    assert manifest_data["quantization_bits"] == 4


def test_run_minigpt_awq_conversion(tiny_minigpt, tmp_path):
    model, config = tiny_minigpt
    out_file = tmp_path / "model.awq.safetensors"
    art, man = run_minigpt_awq_conversion(model, out_file, config, bits=4)
    assert art.is_file()
    assert man.is_file()
    manifest_data = json.loads(man.read_text(encoding="utf-8"))
    assert manifest_data["format"] == "awq"
    assert manifest_data["quantization_bits"] == 4


def test_export_model_gptq_and_awq(tiny_minigpt, tmp_path):
    model, config = tiny_minigpt
    gptq_out = tmp_path / "exported.gptq.safetensors"
    export_model(model, gptq_out, "gptq", model_config=config)
    assert gptq_out.is_file()

    awq_out = tmp_path / "exported.awq.safetensors"
    export_model(model, awq_out, "awq", model_config=config)
    assert awq_out.is_file()
