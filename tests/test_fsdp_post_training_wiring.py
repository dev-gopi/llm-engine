from __future__ import annotations

import json
from pathlib import Path

from post_training.fsdp_post_training import is_fsdp_checkpoint, read_fsdp_manifest


def test_fsdp_manifest_detection(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint.pt"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "gopi-fsdp-post-v2",
                "world_size": 2,
                "metadata": {"training_type": "dpo"},
            }
        ),
        encoding="utf8",
    )
    assert is_fsdp_checkpoint(root)
    assert read_fsdp_manifest(root)["metadata"]["training_type"] == "dpo"


def test_non_fsdp_path_is_not_detected(tmp_path: Path) -> None:
    file_path = tmp_path / "checkpoint.pt"
    file_path.write_bytes(b"not a directory")
    assert not is_fsdp_checkpoint(file_path)


def test_preference_cli_has_fsdp_wiring() -> None:
    text = Path("scripts/train_dpo.py").read_text(encoding="utf8")
    assert '{"ddp", "fsdp", "fsdp_hybrid"}' in text
    assert "save_sharded_post_training" in text
    assert "load_sharded_post_training" in text
    assert "supports distributed_strategy=ddp only" not in text


def test_reward_model_cli_has_fsdp_wiring() -> None:
    text = Path("scripts/train_reward_model.py").read_text(encoding="utf8")
    assert '{"ddp", "fsdp", "fsdp_hybrid"}' in text
    assert "save_sharded_post_training" in text
    assert "load_sharded_post_training" in text
    assert "supports distributed_strategy=ddp only" not in text


def test_offline_grpo_cli_has_fsdp_wiring() -> None:
    text = Path("scripts/train_grpo.py").read_text(encoding="utf8")
    assert '{"ddp", "fsdp", "fsdp_hybrid"}' in text
    assert "save_sharded_post_training" in text
    assert "load_sharded_post_training" in text
    assert "supports distributed_strategy=ddp only" not in text
