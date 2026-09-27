"""Online GRPO with live rollouts, reward scoring, and optional DDP training."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
repository_root = str(Path(__file__).resolve().parents[1])
src_root = str(Path(__file__).resolve().parents[1] / "src")
sys.path[:] = [
    entry
    for entry in sys.path
    if str(Path(entry or ".").resolve())
    not in {script_directory, repository_root, src_root}
]
sys.path[:0] = [src_root, repository_root]

import torch

from inference.generator import Generator
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from optim.adamw import adamw_from_config
from optim.scheduler import Scheduler
from post_training.distributed_online_grpo import aggregate_rollouts, shard_prompts
from post_training.fsdp_post_training import (
    copy_fsdp_weights_to_model,
    is_fsdp_checkpoint,
    load_sharded_post_training,
    save_sharded_post_training,
)
from post_training.grpo import GRPOTrainer
from post_training.grpo_data import build_grpo_loader_from_records
from post_training.online_grpo import (
    OnlineRolloutGenerator,
    RolloutConfig,
    RolloutReplayBuffer,
    load_prompt_records,
    write_rollout_jsonl,
)
from post_training.reward_model import RewardModel
from post_training.reward_scoring import RewardCalibration, RewardScorer
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.distributed import DistributedContext, DistributedTrainer
from utils.config import apply_cli_defaults, load_yaml
from utils.device import resolve_device
from utils.logger import configure_logging, get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model.gpu.yaml")
    )
    parser.add_argument(
        "--training-config", type=Path, default=Path("configs/grpo.online.gpu.yaml")
    )
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path)
    parser.add_argument("--reward-calibration", type=Path)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reward-device", default="cpu")
    args = parser.parse_args()

    config = load_yaml(args.training_config)
    apply_cli_defaults(
        args,
        config.get("runtime", {}),
        {
            "tokenizer": Path("data/tokenizer-finetuning"),
            "output": Path("checkpoints/grpo-online/latest.pt"),
        },
    )
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be used together")
    if not args.resume and args.init_from is None:
        parser.error("--init-from is required for a new online GRPO run")
    configure_logging()
    set_seed(int(config.get("seed", 42)))
    tokenizer = Tokenizer.load(args.tokenizer)
    model_config = adapt_config_to_tokenizer(load_yaml(args.model_config), tokenizer)
    max_length = int(config.get("max_sequence_length", model_config["max_position"]))
    if max_length > int(model_config["max_position"]):
        parser.error("online GRPO max_sequence_length exceeds model context length")
    requested_world_size = int(os.getenv("WORLD_SIZE", "1"))
    strategy = str(config.get("distributed_strategy", "ddp")).lower()
    if strategy not in {"ddp", "fsdp", "fsdp_hybrid"}:
        parser.error(
            "online GRPO distributed_strategy must be ddp, fsdp, or fsdp_hybrid"
        )
    if requested_world_size > 1:
        if strategy.startswith("fsdp"):
            DistributedTrainer.preflight_fsdp()
        distributed = DistributedTrainer.initialize(config.get("distributed_backend"))
        device = distributed.device
    else:
        if strategy.startswith("fsdp"):
            parser.error("FSDP online GRPO requires torchrun with WORLD_SIZE >= 2")
        device = resolve_device(args.device)
        distributed = DistributedContext(
            rank=0, local_rank=0, world_size=1, device=device
        )
    mixed_precision = str(config.get("mixed_precision", "none"))
    if mixed_precision == "fp16" and device.type != "cuda":
        parser.error("online GRPO fp16 requires CUDA; use configs/grpo.online.cpu.yaml")
    if (
        mixed_precision == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error("online GRPO requires BF16, but this GPU does not support it")

    policy = MiniGPT.from_config(model_config, device=device)
    reference = MiniGPT.from_config(model_config, device=device)
    old_policy = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(
        args.reference_checkpoint,
        reference,
        map_location=device,
        use_ema=True,
        restore_rng=False,
        **checkpoint_tokenizer_options(tokenizer),
    )
    fsdp_run = distributed.world_size > 1 and strategy.startswith("fsdp")
    if args.resume and fsdp_run and not is_fsdp_checkpoint(args.resume):
        parser.error(
            "FSDP online GRPO --resume requires a sharded FSDP checkpoint directory; use --init-from for a single-file checkpoint"
        )
    preview = None
    if not args.resume:
        load_checkpoint(
            args.init_from,
            policy,
            map_location=device,
            use_ema=True,
            restore_rng=False,
            **checkpoint_tokenizer_options(tokenizer),
        )
        old_policy.load_state_dict(policy.state_dict())
    elif not fsdp_run:
        preview = load_checkpoint(
            args.resume,
            policy,
            map_location=device,
            restore_rng=False,
            **checkpoint_tokenizer_options(tokenizer),
        )
        if preview.get("metadata", {}).get("training_type") not in {
            None,
            "grpo_online",
        }:
            parser.error("resume checkpoint is not an online GRPO checkpoint")
        old_policy.load_state_dict(policy.state_dict())
    training_policy = DistributedTrainer.wrap(
        policy,
        distributed,
        strategy=strategy if distributed.world_size > 1 else "none",
        mixed_precision=mixed_precision,
    )

    reward_weight = float(config.get("reward_model_weight", 1.0))
    exact_weight = float(config.get("exact_match_weight", 0.0))
    scorer = None
    if reward_weight != 0:
        if args.reward_checkpoint is None:
            parser.error(
                "--reward-checkpoint is required when reward_model_weight is non-zero"
            )
        reward_device = resolve_device(args.reward_device)
        reward_model = RewardModel(
            MiniGPT.from_config(model_config, device=reward_device),
            int(model_config["hidden_size"]),
        ).to(reward_device)
        load_checkpoint(
            args.reward_checkpoint,
            reward_model,
            map_location=reward_device,
            restore_rng=False,
            **checkpoint_tokenizer_options(tokenizer),
        )
        calibration = (
            RewardCalibration.load(args.reward_calibration)
            if args.reward_calibration
            else None
        )
        if bool(config.get("normalize_reward_model", False)) and calibration is None:
            parser.error("normalize_reward_model=true requires --reward-calibration")
        scorer = RewardScorer(
            reward_model, tokenizer, max_length=max_length, calibration=calibration
        )

    prompt_files = config.get("prompt_files", [])
    prompts = load_prompt_records(prompt_files)
    if not prompts:
        parser.error("online GRPO prompt_files contain no usable records")
    iterations = int(config.get("iterations", 1))
    updates_per_rollout = int(config.get("updates_per_rollout", 1))
    max_groups = int(config.get("max_groups_per_iteration", len(prompts)))
    batch_size = int(config.get("batch_size", 1))
    replay_capacity = int(config.get("replay_capacity_groups", 0))
    if iterations < 1 or updates_per_rollout < 1 or max_groups < 1 or batch_size < 1:
        parser.error(
            "iterations, updates_per_rollout, max_groups_per_iteration and batch_size must be positive"
        )

    optimizer = adamw_from_config(training_policy, config)
    estimated_groups = max_groups + (max(0, replay_capacity))
    total_steps = (
        iterations
        * updates_per_rollout
        * max(1, math.ceil(estimated_groups / batch_size))
    )
    scheduler = Scheduler.from_config(optimizer, config, total_steps=total_steps)
    trainer = GRPOTrainer(
        training_policy,
        old_policy,
        reference,
        optimizer,
        scheduler=scheduler,
        clip_epsilon=float(config.get("clip_epsilon", 0.2)),
        beta=float(config.get("beta", 0.04)),
        gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
        mixed_precision=mixed_precision,
        distributed_context=distributed,
    )
    start_iteration = 0
    if args.resume:
        if fsdp_run:
            state = load_sharded_post_training(
                args.resume,
                training_policy,
                optimizer,
                scheduler=scheduler,
                scaler=trainer.scaler,
                map_location="cpu",
            )
        else:
            state = load_checkpoint(
                args.resume,
                policy,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=trainer.scaler,
                map_location=device,
                **checkpoint_tokenizer_options(tokenizer),
            )
        if state.get("metadata", {}).get("training_type") not in {None, "grpo_online"}:
            parser.error("resume checkpoint is not an online GRPO checkpoint")
        trainer.load_state_dict(state.get("trainer", {}))
        start_iteration = int(state.get("metadata", {}).get("online_iteration", 0))

    replay = RolloutReplayBuffer(replay_capacity)
    replay_path = Path(
        config.get("runtime", {}).get(
            "replay_file", "data/processed/grpo/online-replay.jsonl"
        )
    )
    if args.resume and replay_capacity > 0:
        replay.load(replay_path)
    rollout_dir = Path(
        config.get("runtime", {}).get("rollout_dir", "data/processed/grpo/online")
    )
    failure_dir = rollout_dir / "failures"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for iteration in range(start_iteration, iterations):
        if fsdp_run:
            copy_fsdp_weights_to_model(training_policy, old_policy)
        else:
            old_policy.load_state_dict(policy.state_dict())
        local_prompts = shard_prompts(prompts, distributed.rank, distributed.world_size)
        rollout_generator = OnlineRolloutGenerator(
            Generator(old_policy, tokenizer, device=device),
            scorer,
            RolloutConfig(
                group_size=int(config.get("group_size", 4)),
                max_tokens=int(config.get("rollout_max_tokens", 128)),
                temperature=float(config.get("rollout_temperature", 0.8)),
                top_k=int(config.get("rollout_top_k", 40)),
                top_p=float(config.get("rollout_top_p", 1.0)),
                min_p=float(config.get("rollout_min_p", 0.0)),
                repetition_penalty=float(config.get("rollout_repetition_penalty", 1.1)),
                no_repeat_ngram_size=int(config.get("rollout_no_repeat_ngram_size", 3)),
                min_tokens=int(config.get("rollout_min_tokens", 1)),
                reward_model_weight=reward_weight,
                exact_match_weight=exact_weight,
                normalize_reward_model=bool(
                    config.get("normalize_reward_model", False)
                ),
                seed=int(config.get("seed", 42))
                + iteration * 1_000_003
                + distributed.rank * 10_000_019,
            ),
        )
        local_limit = max(1, math.ceil(max_groups / distributed.world_size))
        local_groups, local_failures = rollout_generator.generate(
            local_prompts, max_groups=local_limit
        )
        groups = aggregate_rollouts(local_groups)[:max_groups]
        failures = aggregate_rollouts(local_failures)
        if not groups:
            raise RuntimeError(
                f"online GRPO iteration {iteration + 1} produced no usable groups"
            )
        if distributed.is_main_process:
            write_rollout_jsonl(
                rollout_dir / f"iteration-{iteration + 1:04d}.jsonl", groups
            )
            if failures:
                write_rollout_jsonl(
                    failure_dir / f"iteration-{iteration + 1:04d}.jsonl", failures
                )
        training_records = replay.extend_current(groups)
        if replay_capacity > 0 and distributed.is_main_process:
            replay.save(replay_path)
        loader = build_grpo_loader_from_records(
            training_records,
            tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=True,
            seed=int(config.get("seed", 42)) + iteration,
            num_workers=int(config.get("num_workers", 0)),
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        target_epoch = trainer.current_epoch + updates_per_rollout
        iteration_history = trainer.fit(
            loader,
            epochs=target_epoch,
            validation_loader=None,
            log_every=int(config.get("log_every", 10)),
        )
        metadata = {
            "training_type": "grpo_online",
            "online_iteration": iteration + 1,
            "iterations": iterations,
            "reference_checkpoint": str(args.reference_checkpoint),
            "reward_checkpoint": str(args.reward_checkpoint)
            if args.reward_checkpoint
            else None,
            "reward_calibration": str(args.reward_calibration)
            if args.reward_calibration
            else None,
            "model_config": model_config,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "replay_file": str(replay_path) if replay_capacity > 0 else None,
            "distributed_world_size": distributed.world_size,
        }
        if fsdp_run:
            save_sharded_post_training(
                args.output,
                training_policy,
                optimizer,
                metadata=metadata,
                trainer_state=trainer.state_dict(),
                scheduler=scheduler,
                scaler=trainer.scaler,
            )
        elif distributed.is_main_process:
            save_checkpoint(
                args.output,
                policy,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=trainer.scaler,
                step=trainer.global_step,
                trainer=trainer.state_dict(),
                metadata=metadata,
            )
        DistributedTrainer.barrier(distributed)
        record = {
            "iteration": iteration + 1,
            "groups": len(groups),
            "failures": len(failures),
            "training_groups": len(training_records),
            "step": trainer.global_step,
            "history": iteration_history[-updates_per_rollout:],
        }
        history.append(record)
        logger.info(
            "online_grpo iteration=%d groups=%d failures=%d step=%d",
            iteration + 1,
            len(groups),
            len(failures),
            trainer.global_step,
        )

    if distributed.is_main_process:
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output),
                    "training_type": "grpo_online",
                    "iterations_completed": iterations,
                    "world_size": distributed.world_size,
                    "step": trainer.global_step,
                    "history": history,
                },
                indent=2,
            )
        )
    DistributedTrainer.barrier(distributed)
    DistributedTrainer.shutdown()


if __name__ == "__main__":
    main()
