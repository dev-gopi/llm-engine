from argparse import Namespace
from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).parents[1] / "scripts" / "serve_gguf.py"
SPEC = importlib.util.spec_from_file_location("serve_gguf", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_gguf_launcher_builds_explicit_resource_command(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    args = Namespace(
        llama_server="llama-server", model=str(model), host="127.0.0.1", port=8080,
        context=8192, parallel=2, gpu_layers=20, cache_type_k="q8_0", cache_type_v="q8_0",
        flash_attention=True,
    )
    cmd = MODULE.command(args)
    assert "--model" in cmd and str(model.resolve()) in cmd
    assert cmd[cmd.index("--ctx-size") + 1] == "8192"
    assert cmd[cmd.index("--parallel") + 1] == "2"
    assert "--flash-attn" in cmd
