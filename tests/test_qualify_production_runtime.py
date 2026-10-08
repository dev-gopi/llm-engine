import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_production_runtime.py"
PROFILE = ROOT / "configs" / "production.moe.large.yaml"


def test_qualification_cli_runs_without_shadowing_stdlib_tokenize(tmp_path) -> None:
    output = tmp_path / "qualification.json"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(PROFILE), "--output", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["profiles"][0]["profile"] == "100b_sparse_moe"
    assert report["all_qualified"] is False
