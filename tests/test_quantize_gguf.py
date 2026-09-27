import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "quantize_gguf.py"
SPEC = importlib.util.spec_from_file_location("quantize_gguf", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_quantize_gguf_command_and_validation(tmp_path):
    source = tmp_path / "model-f16.gguf"
    source.write_bytes(b"GGUF" + b"\0" * 32)
    output = tmp_path / "model-q4.gguf"
    command = MODULE.build_command(source, output, "q4_k_m")
    assert command[-1] == "Q4_K_M"
    assert command[-3] == str(source.resolve())
    assert command[-2] == str(output.resolve())

    output.write_bytes(b"GGUF" + b"\0" * 32)
    assert MODULE.validate_gguf(output) == output


def test_quantize_gguf_rejects_unsafe_inputs(tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"GGUF" + b"\0" * 32)
    with pytest.raises(ValueError, match="differ"):
        MODULE.build_command(source, source, "Q4_K_M")
    with pytest.raises(ValueError, match="unsupported quantization type"):
        MODULE.build_command(source, tmp_path / "out.gguf", "made_up")
