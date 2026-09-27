"""Train a policy with Direct Preference Optimization on chosen/rejected pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
repository_root = str(Path(__file__).resolve().parents[1])
src_root = str(Path(__file__).resolve().parents[1] / "src")
# Keep the engine source root ahead of site-packages.  ``datasets`` otherwise
# resolves to the optional Hugging Face distribution during direct script
# launches, even if an editable installation has already listed ``src``.
sys.path[:] = [
    entry
    for entry in sys.path
    if str(Path(entry or ".").resolve())
    not in {script_directory, repository_root, src_root}
]
sys.path[:0] = [src_root, repository_root]

import torch

from local_dataset.governance import enforce_dataset_governance
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from optim.adamw import adamw_from_config
from optim.scheduler import Scheduler
from post_training.dpo import DPOTrainer
from post_training.fsdp_post_training import (
    is_fsdp_checkpoint,
    load_sharded_post_training,
    save_sharded_post_training,
)
from post_training.preference_data import build_kto_loader, build_preference_loader
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
        "--training-config", type=Path, default=Path("configs/dpo.gpu.yaml")
    )
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--best-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--method", choices=("dpo", "ipo", "orpo", "kto"), default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_yaml(args.training_config)
    method = str(args.method or config.get("method", "dpo")).lower()
    if method in {"dpo", "ipo", "kto"} and args.reference_checkpoint is None:
        parser.error(f"--reference-checkpoint is required for {method.upper()}")
    if (
        method == "orpo"
        and args.init_from is None
        and args.reference_checkpoint is None
        and args.resume is None
    ):
        parser.error("ORPO requires --init-from (or --resume) to initialize the policy")
    apply_cli_defaults(
        args,
        config.get("runtime", {}),
        {
            "tokenizer": Path("data/tokenizer-finetuning"),
            "output": Path("checkpoints/dpo/latest.pt"),
            "best_output": Path("checkpoints/dpo/best.pt"),
        },
    )
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be used together")
    configure_logging()
    model_config = load_yaml(args.model_config)
    set_seed(int(config.get("seed", 42)))
    paths = [*config.get("train_files", []), *config.get("validation_files", [])]
    for finding in enforce_dataset_governance(paths, config.get("dataset_governance")):
        logger.warning("dataset governance [%s]: %s", finding.code, finding.message)
    requested_world_size = int(os.getenv("WORLD_SIZE", "1"))
    strategy = str(config.get("distributed_strategy", "ddp")).lower()
    if strategy not in {"ddp", "fsdp", "fsdp_hybrid"}:
        parser.error(
            "preference training distributed_strategy must be ddp, fsdp, or fsdp_hybrid"
        )
    if requested_world_size > 1:
        if strategy.startswith("fsdp"):
            DistributedTrainer.preflight_fsdp()
        distributed = DistributedTrainer.initialize(config.get("distributed_backend"))
        device = distributed.device
    else:
        if strategy.startswith("fsdp"):
            parser.error(
                "FSDP preference training requires torchrun with WORLD_SIZE >= 2"
            )
        device = resolve_device(args.device)
        distributed = DistributedContext(
            rank=0, local_rank=0, world_size=1, device=device
        )
    tracker = create_tracker_from_config(
        config, is_main_process=distributed.is_main_process
    )

    mixed_precision = str(config.get("mixed_precision", "none"))
    if mixed_precision == "fp16" and device.type != "cuda":
        parser.error(
            "the selected DPO profile requires CUDA fp16; use configs/dpo.cpu.yaml"
        )
    if (
        mixed_precision == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error(
            "the selected DPO profile requires BF16, but this GPU does not support it"
        )
    tokenizer = Tokenizer.load(args.tokenizer)
    try:
        model_config = adapt_config_to_tokenizer(model_config, tokenizer)
    except ValueError as error:
        parser.error(str(error))
    if int(config["max_sequence_length"]) > int(model_config["max_position"]):
        parser.error("DPO max_sequence_length exceeds the model context length")

    policy = MiniGPT.from_config(model_config, device=device)
    reference = None
    if method in {"dpo", "ipo", "kto"}:
        reference = MiniGPT.from_config(model_config, device=device)
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
            "FSDP preference --resume requires a sharded FSDP checkpoint directory; use --init-from for a single-file checkpoint"
        )
    if not args.resume:
        load_checkpoint(
            args.init_from or args.reference_checkpoint,
            policy,
            map_location=device,
            use_ema=True,
            restore_rng=False,
            **checkpoint_tokenizer_options(tokenizer),
        )
    elif not fsdp_run:
        load_checkpoint(
            args.resume,
            policy,
            map_location=device,
            restore_rng=False,
            **checkpoint_tokenizer_options(tokenizer),
        )
    training_policy = DistributedTrainer.wrap(
        policy,
        distributed,
        strategy=strategy if distributed.world_size > 1 else "none",
        mixed_precision=mixed_precision,
    )
    loader_builder = build_kto_loader if method == "kto" else build_preference_loader
    train_loader = loader_builder(
        config["train_files"],
        tokenizer,
        max_length=int(config["max_sequence_length"]),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        seed=int(config.get("seed", 42)),
        num_workers=int(config.get("num_workers", 0)),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    validation_loader = (
        loader_builder(
            config["validation_files"],
            tokenizer,
            max_length=int(config["max_sequence_length"]),
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        if config.get("validation_files")
        else None
    )
    epochs = args.epochs or int(config.get("epochs", 1))
    optimizer = adamw_from_config(training_policy, config)
    scheduler = Scheduler.from_config(
        optimizer, config, total_steps=max(1, len(train_loader) * epochs)
    )
    trainer = DPOTrainer(
        training_policy,
        reference,
        optimizer,
        beta=float(config.get("beta", 0.1)),
        label_smoothing=float(config.get("label_smoothing", 0.0)),
        orpo_lambda=float(config.get("orpo_lambda", 0.1)),
        kto_desirable_weight=float(config.get("kto_desirable_weight", 1.0)),
        kto_undesirable_weight=float(config.get("kto_undesirable_weight", 1.0)),
        scheduler=scheduler,
        gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
        mixed_precision=mixed_precision,
        method=method,
        distributed_context=distributed,
    )
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
        if state.get("metadata", {}).get("training_type") not in {None, method}:
            parser.error(f"resume checkpoint training_type does not match {method}")
        trainer.load_state_dict(state.get("trainer", {}))

    def save(
        path: Path, current: DPOTrainer, epoch: int, *, best: bool = False
    ) -> None:
        metadata = {
            "epoch": epoch + 1,
            "best": best,
            "model_config": model_config,
            "reference_checkpoint": str(args.reference_checkpoint)
            if args.reference_checkpoint
            else None,
            "training_type": method,
            "distributed_strategy": strategy if distributed.world_size > 1 else "none",
            "distributed_world_size": distributed.world_size,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "step": current.global_step,
        }
        if fsdp_run:
            save_sharded_post_training(
                path,
                training_policy,
                optimizer,
                metadata=metadata,
                trainer_state=current.state_dict(),
                scheduler=scheduler,
                scaler=current.scaler,
            )
            return
        if not distributed.is_main_process:
            return
        save_checkpoint(
            path,
            policy,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=current.scaler,
            step=current.global_step,
            trainer=current.state_dict(),
            metadata=metadata,
        )

    history = trainer.fit(
        train_loader,
        epochs=epochs,
        validation_loader=validation_loader,
        checkpoint_callback=lambda current, epoch: save(args.output, current, epoch),
        best_checkpoint_callback=lambda current, epoch: save(
            args.best_output, current, epoch, best=True
        ),
        early_stopping_patience=config.get("early_stopping_patience"),
        log_every=int(config.get("log_every", 10)),
    )
    if distributed.is_main_process:
        log_history(tracker, history, prefix="post_training")
        tracker.close()
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output),
                    "best_checkpoint": str(args.best_output),
                    "method": method,
                    "world_size": distributed.world_size,
                    "step": trainer.global_step,
                    "stopped_early": trainer.stopped_early,
                    "history": history,
                },
                indent=2,
            )
        )
    DistributedTrainer.barrier(distributed)
    DistributedTrainer.shutdown()


if __name__ == "__main__":
    main()
