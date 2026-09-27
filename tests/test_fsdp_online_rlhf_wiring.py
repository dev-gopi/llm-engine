from __future__ import annotations

from pathlib import Path


def test_ppo_cli_has_fsdp_rlhf_checkpoint_wiring() -> None:
    source = Path("scripts/train_ppo.py").read_text(encoding="utf8")
    assert '{"none", "ddp", "fsdp", "fsdp_hybrid"}' in source
    assert "save_sharded_rlhf_checkpoint" in source
    assert "load_sharded_rlhf_checkpoint" in source
    assert "copy_fsdp_weights_to_model" in source
    assert (
        "distributed PPO requires at least one usable rollout on every rank" in source
    )
    assert Path("configs/ppo.fsdp.yaml").is_file()
    assert "distributed_strategy: fsdp" in Path("configs/ppo.fsdp.yaml").read_text(
        encoding="utf8"
    )


def test_online_grpo_cli_has_fsdp_wiring() -> None:
    source = Path("scripts/train_grpo_online.py").read_text(encoding="utf8")
    assert '{"ddp", "fsdp", "fsdp_hybrid"}' in source
    assert "save_sharded_post_training" in source
    assert "load_sharded_post_training" in source
    assert "copy_fsdp_weights_to_model" in source
    assert "supports distributed_strategy=ddp only" not in source
    assert Path("configs/grpo.online.fsdp.yaml").is_file()
    assert "distributed_strategy: fsdp" in Path(
        "configs/grpo.online.fsdp.yaml"
    ).read_text(encoding="utf8")


def test_fsdp_replica_copy_helper_uses_collective_full_state() -> None:
    source = Path("src/post_training/fsdp_post_training.py").read_text(encoding="utf8")
    assert "def copy_fsdp_weights_to_model" in source
    assert "FullStateDictConfig" in source
    assert "rank0_only=False" in source


def test_ppo_and_grpo_use_fsdp_safe_gradient_clipping() -> None:
    ppo = Path("src/post_training/ppo.py").read_text(encoding="utf8")
    grpo = Path("src/post_training/grpo.py").read_text(encoding="utf8")
    assert 'getattr(self.actor, "clip_grad_norm_", None)' in ppo
    assert 'getattr(self.value_model, "clip_grad_norm_", None)' in ppo
    assert 'getattr(self.policy, "clip_grad_norm_", None)' in grpo
