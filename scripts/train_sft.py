"""Convenience entry point for supervised fine-tuning.

This wrapper intentionally delegates to the production training engine so SFT
inherits the same DDP/FSDP, resume, checkpoint, reporting, PEFT, and validation
behavior as ``scripts/train.py``. It only supplies the standard SFT profile
when the caller does not provide ``--training-config``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
for entry in (str(SRC_ROOT), str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _inject_default_sft_profile(argv: list[str]) -> list[str]:
    if "--training-config" in argv:
        return argv
    return [*argv, "--training-config", "configs/finetuning.gpu.yaml"]


def main() -> None:
    sys.argv = _inject_default_sft_profile(sys.argv)
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
