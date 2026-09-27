"""Train a policy with online sequence-level PPO RLHF."""

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
    e
    for e in sys.path
    if str(Path(e or ".").resolve())
    not in {script_directory, repository_root, src_root}
]
sys.path[:0] = [src_root, repository_root]

import torch

from inference.generator import Generator
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from optim.adamw import adamw_from_config
from optim.scheduler import Scheduler
from post_training.distributed_online_grpo import shard_prompts
from post_training.fsdp_post_training import (
    copy_fsdp_weights_to_model,
    is_fsdp_checkpoint,
    load_sharded_rlhf_checkpoint,
    save_sharded_rlhf_checkpoint,
)
from post_training.online_grpo import load_prompt_records, write_rollout_jsonl
from post_training.ppo import (
    AdaptiveKLController,
    PPOTrainer,
    ValueModel,
    load_value_checkpoint,
    save_value_checkpoint,
)
from post_training.ppo_online import collect_ppo_rollouts
from post_training.reward_model import RewardModel
from post_training.reward_scoring import RewardCalibration, RewardScorer
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.distributed import DistributedContext, DistributedTrainer
from training.experiment_tracking import create_tracker_from_config, log_history
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
        "--training-config", type=Path, default=Path("configs/ppo.gpu.yaml")
    )
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path)
    parser.add_argument("--reward-calibration", type=Path)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--value-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--value-output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reward-device", default="cpu")
    args = parser.parse_args()

    config = load_yaml(args.training_config)
    apply_cli_defaults(
        args,
        config.get("runtime", {}),
        {
            "tokenizer": Path("data/tokenizer-finetuning"),
            "output": Path("checkpoints/ppo/latest.pt"),
            "value_output": Path("checkpoints/ppo/value-latest.pt"),
        },
    )
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be used together")
    if not args.resume and args.init_from is None:
        parser.error("--init-from is required for a new PPO run")
    configure_logging()
    set_seed(int(config.get("seed", 42)))
    requested_world_size = int(os.getenv("WORLD_SIZE", "1"))
    strategy = str(config.get("distributed_strategy", "none")).lower()
    if strategy not in {"none", "ddp", "fsdp", "fsdp_hybrid"}:
        parser.error("PPO distributed_strategy must be none, ddp, fsdp, or fsdp_hybrid")
    if requested_world_size > 1:
        if strategy == "none":
            parser.error(
                "PPO WORLD_SIZE > 1 requires distributed_strategy=ddp, fsdp, or fsdp_hybrid"
            )
        if strategy.startswith("fsdp"):
            DistributedTrainer.preflight_fsdp()
        distributed = DistributedTrainer.initialize(config.get("distributed_backend"))
        device = distributed.device
    else:
        if strategy.startswith("fsdp"):
            parser.error("FSDP PPO requires torchrun with WORLD_SIZE >= 2")
        device = resolve_device(args.device)
        distributed = DistributedContext(
            rank=0, local_rank=0, world_size=1, device=device
        )
    tracker = create_tracker_from_config(
        config, is_main_process=distributed.is_main_process
    )

    mixed_precision = str(config.get("mixed_precision", "none"))
    if mixed_precision != "none":
        parser.error(
            "PPO currently uses explicit fp32 optimizer steps; set mixed_precision: none"
        )

    tokenizer = Tokenizer.load(args.tokenizer)
    model_config = adapt_config_to_tokenizer(load_yaml(args.model_config), tokenizer)
    max_length = int(config.get("max_sequence_length", model_config["max_position"]))
    if max_length > int(model_config["max_position"]):
        parser.error("PPO max_sequence_length exceeds model context length")

    actor = MiniGPT.from_config(model_config, device=device)
    reference = MiniGPT.from_config(model_config, device=device)
    rollout_policy = MiniGPT.from_config(model_config, device=device)
    value_model = ValueModel(
        MiniGPT.from_config(model_config, device=device),
        int(model_config["hidden_size"]),
    ).to(device)
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
            "FSDP PPO --resume requires a sharded RLHF checkpoint directory; use --init-from for a single-file checkpoint"
        )
    if not args.resume:
        load_checkpoint(
            args.init_from,
            actor,
            map_location=device,
            use_ema=True,
            restore_rng=False,
            **checkpoint_tokenizer_options(tokenizer),
        )
        rollout_policy.load_state_dict(actor.state_dict())
        value_model.backbone.load_state_dict(actor.state_dict())
        if args.value_checkpoint:
            load_value_checkpoint(
                args.value_checkpoint, value_model, map_location=device
            )
    elif not fsdp_run:
        state = load_checkpoint(
            args.resume,
            actor,
            map_location=device,
            restore_rng=False,
            **checkpoint_tokenizer_options(tokenizer),
        )
        if state.get("metadata", {}).get("training_type") not in {None, "ppo"}:
            parser.error("resume checkpoint is not a PPO checkpoint")
        rollout_policy.load_state_dict(actor.state_dict())
        value_model.backbone.load_state_dict(actor.state_dict())
        value_path = args.value_checkpoint or (
            Path(state.get("metadata", {}).get("value_checkpoint"))
            if state.get("metadata", {}).get("value_checkpoint")
            else None
        )
        if value_path is not None:
            load_value_checkpoint(value_path, value_model, map_location=device)

    training_actor = DistributedTrainer.wrap(
        actor,
        distributed,
        strategy=strategy if distributed.world_size > 1 else "none",
        mixed_precision="none",
    )
    training_value = DistributedTrainer.wrap(
        value_model,
        distributed,
        strategy=strategy if distributed.world_size > 1 else "none",
        mixed_precision="none",
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
        scorer = RewardScorer(
            reward_model, tokenizer, max_length=max_length, calibration=calibration
        )
    if reward_weight == 0 and exact_weight == 0:
        parser.error("PPO needs reward_model_weight or exact_match_weight")

    prompts = load_prompt_records(config.get("prompt_files", []))
    if not prompts:
        parser.error("PPO prompt_files contain no usable records")
    iterations = int(config.get("iterations", 1))
    rollouts_per_iteration = int(config.get("rollouts_per_iteration", len(prompts)))
    minibatch_size = int(config.get("minibatch_size", config.get("batch_size", 1)))
    ppo_epochs = int(config.get("ppo_epochs", 1))
    if min(iterations, rollouts_per_iteration, minibatch_size, ppo_epochs) < 1:
        parser.error("PPO iteration/batch settings must be positive")

    optimizer = adamw_from_config(training_actor, config)
    value_config = dict(config)
    value_config["learning_rate"] = float(
        config.get("value_learning_rate", config.get("learning_rate", 1e-5))
    )
    value_optimizer = adamw_from_config(training_value, value_config)
    total_updates = max(
        1,
        iterations
        * ppo_epochs
        * ((rollouts_per_iteration + minibatch_size - 1) // minibatch_size),
    )
    scheduler = Scheduler.from_config(optimizer, config, total_steps=total_updates)
    value_scheduler = Scheduler.from_config(
        value_optimizer, value_config, total_steps=total_updates
    )
    kl_controller = AdaptiveKLController(
        coefficient=float(config.get("kl_coefficient", 0.04)),
        target=float(config.get("kl_target", 0.1)),
        horizon=int(config.get("kl_horizon", 10000)),
    )
    trainer = PPOTrainer(
        training_actor,
        training_value,
        reference,
        reward_model if reward_weight != 0 else reference,
        optimizer,
        value_optimizer=value_optimizer,
        scheduler=scheduler,
        value_scheduler=value_scheduler,
        clip_epsilon=float(config.get("clip_epsilon", 0.2)),
        value_clip_epsilon=config.get("value_clip_epsilon", 0.2),
        gamma=float(config.get("gamma", 1.0)),
        lam=float(config.get("gae_lambda", 1.0)),
        kl_controller=kl_controller,
        value_coef=float(config.get("value_coef", 0.5)),
        entropy_coef=float(config.get("entropy_coef", 0.0)),
        grad_clip=config.get("gradient_clip_norm", 1.0),
    )
    start_iteration = 0
    if args.resume:
        if fsdp_run:
            state = load_sharded_rlhf_checkpoint(
                args.resume,
                {"actor": training_actor, "value": training_value},
                {"actor": optimizer, "value": value_optimizer},
                map_location="cpu",
            )
            metadata = state.get("metadata", {})
            extra = state.get("extra_state", {})
            if extra.get("scheduler") is not None:
                scheduler.load_state_dict(extra["scheduler"])
            if extra.get("value_scheduler") is not None:
                value_scheduler.load_state_dict(extra["value_scheduler"])
            trainer.global_step = int(
                extra.get("ppo_global_step", metadata.get("ppo_global_step", 0))
            )
            kl_controller.coefficient = float(
                extra.get(
                    "kl_coefficient",
                    metadata.get("kl_coefficient", kl_controller.coefficient),
                )
            )
        else:
            state = load_checkpoint(
                args.resume,
                actor,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
                **checkpoint_tokenizer_options(tokenizer),
            )
            metadata = state.get("metadata", {})
            trainer.global_step = int(
                metadata.get("ppo_global_step", state.get("step", 0))
            )
            kl_controller.coefficient = float(
                metadata.get("kl_coefficient", kl_controller.coefficient)
            )
            value_path = args.value_checkpoint or (
                Path(metadata["value_checkpoint"])
                if metadata.get("value_checkpoint")
                else None
            )
            if value_path is not None:
                load_value_checkpoint(
                    value_path, value_model, value_optimizer, map_location=device
                )
        if metadata.get("training_type") not in {None, "ppo"}:
            parser.error("resume checkpoint is not a PPO checkpoint")
        start_iteration = int(metadata.get("iteration", 0))

    rollout_dir = Path(
        config.get("runtime", {}).get("rollout_dir", "data/processed/ppo/online")
    )
    rollout_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for iteration in range(start_iteration, iterations):
        if fsdp_run:
            copy_fsdp_weights_to_model(training_actor, rollout_policy)
        else:
            rollout_policy.load_state_dict(actor.state_dict())
        local_prompts = (
            shard_prompts(prompts, distributed.rank, distributed.world_size)
            if distributed.world_size > 1
            else prompts
        )
        if not local_prompts:
            # Keep every distributed rank active even when prompts < world size.
            local_prompts = [prompts[distributed.rank % len(prompts)]]
        local_target = max(
            1, math.ceil(rollouts_per_iteration / distributed.world_size)
        )
        rollout, failures = collect_ppo_rollouts(
            local_prompts,
            generator=Generator(rollout_policy, tokenizer, device=device),
            tokenizer=tokenizer,
            old_policy=rollout_policy,
            value_model=training_value,
            scorer=scorer,
            device=device,
            max_sequence_length=max_length,
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
            normalize_reward_model=bool(config.get("normalize_reward_model", False)),
            seed=int(config.get("seed", 42))
            + iteration * 1_000_003
            + distributed.rank * 10_000_019,
            max_rollouts=local_target,
        )
        if distributed.world_size > 1:
            import torch.distributed as dist

            counts = [None for _ in range(distributed.world_size)]
            dist.all_gather_object(counts, int(rollout["token_ids"].shape[0]))
            common = min(int(count) for count in counts)
            if common < 1:
                raise RuntimeError(
                    "distributed PPO requires at least one usable rollout on every rank"
                )
            rollout = {key: value[:common] for key, value in rollout.items()}
        if (
            bool(config.get("normalize_advantages", True))
            and rollout["advantages"].numel() > 1
        ):
            adv = rollout["advantages"]
            rollout["advantages"] = (adv - adv.mean()) / adv.std(
                unbiased=False
            ).clamp_min(1e-6)
        metrics = trainer.update(
            rollout,
            minibatch_size=min(minibatch_size, int(rollout["token_ids"].shape[0])),
            epochs=ppo_epochs,
        )
        local_rollouts = int(rollout["token_ids"].shape[0])
        local_failures = len(failures)
        if distributed.world_size > 1:
            import torch.distributed as dist

            metric_keys = ("loss", "policy_loss", "value_loss", "kl")
            packed = torch.tensor(
                [
                    *(float(metrics[key]) for key in metric_keys),
                    float(local_rollouts),
                    float(local_failures),
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            for index, key in enumerate(metric_keys):
                metrics[key] = float(packed[index].item() / distributed.world_size)
            global_rollouts = int(packed[-2].item())
            global_failures = int(packed[-1].item())
        else:
            global_rollouts, global_failures = local_rollouts, local_failures
        record = {
            "iteration": iteration + 1,
            "rollouts": global_rollouts,
            "failures": global_failures,
            **metrics,
        }
        history.append(record)
        if failures and distributed.is_main_process:
            write_rollout_jsonl(
                rollout_dir / f"failures-{iteration + 1:04d}.jsonl", failures
            )
        metadata = {
            "training_type": "ppo",
            "iteration": iteration + 1,
            "ppo_global_step": trainer.global_step,
            "kl_coefficient": kl_controller.coefficient,
            "reference_checkpoint": str(args.reference_checkpoint),
            "reward_checkpoint": str(args.reward_checkpoint)
            if args.reward_checkpoint
            else None,
            "model_config": model_config,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "distributed_strategy": strategy if distributed.world_size > 1 else "none",
            "distributed_world_size": distributed.world_size,
        }
        if fsdp_run:
            save_sharded_rlhf_checkpoint(
                args.output,
                {"actor": training_actor, "value": training_value},
                {"actor": optimizer, "value": value_optimizer},
                metadata=metadata,
                extra_state={
                    "ppo_global_step": trainer.global_step,
                    "kl_coefficient": kl_controller.coefficient,
                    "scheduler": scheduler.state_dict(),
                    "value_scheduler": value_scheduler.state_dict(),
                },
            )
        elif distributed.is_main_process:
            value_path = save_value_checkpoint(
                args.value_output,
                value_model,
                value_optimizer,
                step=trainer.global_step,
                metadata={"training_type": "ppo_value", "iteration": iteration + 1},
            )
            metadata["value_checkpoint"] = str(value_path)
            save_checkpoint(
                args.output,
                actor,
                optimizer=optimizer,
                scheduler=scheduler,
                step=trainer.global_step,
                metadata=metadata,
            )
        DistributedTrainer.barrier(distributed)
        if distributed.is_main_process:
            logger.info(
                "ppo iteration=%d rollouts=%d loss=%.6f kl=%.6f",
                iteration + 1,
                record["rollouts"],
                metrics["loss"],
                metrics["kl"],
            )

    if distributed.is_main_process:
        log_history(tracker, history, prefix="post_training")
        tracker.close()
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output),
                    "value_checkpoint": None if fsdp_run else str(args.value_output),
                    "training_type": "ppo",
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
