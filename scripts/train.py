"""Train or resume the Gopi causal language model."""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from model.gpt import MiniGPT
from datasets.governance import enforce_dataset_governance
from model.loss import CausalLanguageModelLoss
from optim.adamw import adamw_from_config
from optim.ema import EMA
from optim.scheduler import Scheduler
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.distributed_checkpoint import load_distributed_checkpoint, save_distributed_checkpoint
from training.data import build_loader
from training.distributed import DistributedTrainer
from training.evaluator import Evaluator
from training.trainer import Trainer
from training.planner import optimizer_steps_for_epochs
from training.elastic import PreemptionCoordinator
from dotenv import load_dotenv

from utils.config import load_yaml
from utils.logger import configure_logging, get_logger
from utils.seed import set_seed

load_dotenv()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.v2.gpu.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/v2-training/latest.pt"))
    parser.add_argument("--best-output", type=Path, default=Path("checkpoints/v2-training/best.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path, help="load model weights only for a new training stage")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be used together")

    configure_logging()
    model_config, config = load_yaml(args.model_config), load_yaml(args.training_config)
    precision = str(config.get("mixed_precision", "none"))
    if precision == "fp16" and not torch.cuda.is_available():
        parser.error(
            "the selected GPU training profile requires CUDA, but PyTorch cannot access a GPU. "
            "Fix the NVIDIA driver until `nvidia-smi` works, or explicitly select "
            "--model-config configs/model.v2.cpu.yaml --training-config configs/training.cpu.yaml"
        )
    if precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        parser.error(
            "the selected profile requires CUDA BF16, but this GPU does not support it; "
            "use an FP16 profile or set mixed_precision: fp16"
        )
    base_seed = int(config.get("seed", 42))
    set_seed(base_seed)
    governed_paths = [*config.get("train_files", []), *config.get("validation_files", [])]
    governance_findings = enforce_dataset_governance(
        governed_paths, config.get("dataset_governance"),
    )
    for finding in governance_findings:
        logger.warning("dataset governance [%s]: %s", finding.code, finding.message)
    distributed = DistributedTrainer.initialize(config.get("distributed_backend"))
    atexit.register(DistributedTrainer.shutdown)
    tokenizer = Tokenizer.load(args.tokenizer)
    configured_vocab_size = int(model_config["vocab_size"])
    if tokenizer.vocab_size != configured_vocab_size:
        if (
            tokenizer.base_vocab_size == configured_vocab_size
            and tokenizer.vocab_size > configured_vocab_size
        ):
            model_config = dict(model_config)
            model_config["vocab_size"] = tokenizer.vocab_size
            logger.info(
                "Using append-only tokenizer extension: resizing vocabulary from %d to %d",
                configured_vocab_size, tokenizer.vocab_size,
            )
        else:
            parser.error(
                "tokenizer vocabulary does not match model vocab_size and is not a "
                "verified append-only extension of that vocabulary"
            )
    max_seq_len = int(config.get("max_sequence_length", 0))
    max_pos = int(model_config.get("max_position", 0))
    if max_seq_len > max_pos:
        parser.error(
            f"training max_sequence_length ({max_seq_len}) exceeds model max_position ({max_pos}). "
            f"Use a matching training config (e.g. max_sequence_length <= {max_pos}) or a model config with max_position >= {max_seq_len}."
        )
    model = MiniGPT.from_config(model_config, device=distributed.device)
    if args.init_from:
        # A new training stage should start from the learned model itself. EMA
        # can lag badly during short stages and is recreated for this run below.
        load_checkpoint(
            args.init_from,
            model,
            map_location=distributed.device,
            use_ema=False,
            restore_rng=False,
            expected_tokenizer_fingerprint=tokenizer.fingerprint,
            compatible_tokenizer_fingerprints=tokenizer.compatible_base_fingerprints,
            allow_vocab_extension=bool(tokenizer.compatible_base_fingerprints),
        )
    strategy = str(config.get("distributed_strategy", "ddp"))
    distributed_checkpoints = strategy.startswith("fsdp") or str(
        config.get("checkpoint_format", "single_file")
    ).lower() == "distributed"
    training_model = DistributedTrainer.wrap(
        model,
        distributed,
        strategy=strategy,
        mixed_precision=str(config.get("mixed_precision", "none")),
    )
    # Model initialization must be identical across ranks, but stochastic
    # training operations (for example dropout) should not reuse identical RNG
    # streams on every worker.
    if distributed.world_size > 1 and not args.resume:
        set_seed(base_seed + distributed.rank)
    train_loader = build_loader(config["train_files"], tokenizer, config, shuffle=True, rank=distributed.rank, world_size=distributed.world_size)
    epochs = args.epochs or int(config.get("epochs", 1))
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    total_steps = optimizer_steps_for_epochs(len(train_loader), epochs, accumulation)
    optimizer = adamw_from_config(training_model, config)
    scheduler = Scheduler.from_config(optimizer, config, total_steps=total_steps)
    # A conventional EMA duplicates every parameter and defeats FSDP memory
    # sharding. Large FSDP jobs should average selected exported checkpoints.
    ema = None if strategy.startswith("fsdp") else EMA(
        training_model, decay=float(config.get("ema_decay", 0.999))
    )
    loss_fn = CausalLanguageModelLoss.from_config(config)
    trainer = Trainer(
        training_model, optimizer, loss_fn, scheduler=scheduler, ema=ema,
        gradient_clip_norm=config.get("gradient_clip_norm", 1.0), device=distributed.device,
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(config.get("mixed_precision", "none")),
        grad_scaler_initial_scale=float(config.get("grad_scaler_initial_scale", 65536.0)),
        grad_scaler_growth_interval=int(config.get("grad_scaler_growth_interval", 2000)),
    )
    preemption = PreemptionCoordinator()
    preemption.install()
    atexit.register(preemption.restore)
    if args.resume:
        if args.resume.is_dir():
            state = load_distributed_checkpoint(
                args.resume, training_model, optimizer,
                scheduler=scheduler, scaler=trainer.scaler,
            )
            saved_fingerprint = state.get("tokenizer_fingerprint")
            if saved_fingerprint and saved_fingerprint != tokenizer.fingerprint:
                parser.error(
                    "checkpoint tokenizer fingerprint does not match the selected tokenizer"
                )
        else:
            state = load_checkpoint(
                args.resume, model, optimizer=optimizer, scheduler=scheduler,
                ema=ema, scaler=trainer.scaler, map_location=distributed.device,
                expected_tokenizer_fingerprint=tokenizer.fingerprint,
            )
        trainer.global_step = state["step"]
        trainer.load_state_dict(state.get("trainer", {}))
        if distributed.world_size > 1 and not distributed_checkpoints:
            # A single-file DDP checkpoint contains rank-zero RNG only. Avoid
            # cloning that stream across all workers after resume.
            set_seed(base_seed + distributed.rank + trainer.global_step)
        # Checkpoints contain the scaler's old tuning. Keep the current run's
        # configured growth interval when resuming so stability changes apply.
        if trainer.scaler.is_enabled():
            trainer.scaler.set_growth_interval(
                int(config.get("grad_scaler_growth_interval", 2000))
            )
        sampler_state = state.get("sampler")
        if sampler_state and hasattr(train_loader.batch_sampler, "load_state_dict"):
            train_loader.batch_sampler.load_state_dict(sampler_state)

    validation_loader = None
    evaluator = None
    if config.get("validation_files"):
        validation_loader = build_loader(config["validation_files"], tokenizer, config, shuffle=False, rank=distributed.rank, world_size=distributed.world_size)
        evaluator = Evaluator(training_model, loss_fn=loss_fn, device=distributed.device)

    def checkpoint_callback(current: Trainer, epoch: int) -> None:
        if distributed_checkpoints:
            save_distributed_checkpoint(
                args.output,
                training_model,
                optimizer,
                scheduler=scheduler,
                scaler=current.scaler,
                metadata={
                    "step": current.global_step,
                    "epoch": epoch + 1,
                    "model_config": model_config,
                    "tokenizer_fingerprint": tokenizer.fingerprint,
                    "trainer": current.state_dict(),
                    "sampler": train_loader.batch_sampler.state_dict(),
                },
            )
            return
        if not distributed.is_main_process:
            return
        save_checkpoint(args.output, model, optimizer=optimizer, scheduler=scheduler, ema=ema, scaler=current.scaler,
                        step=current.global_step, metadata={"epoch": epoch + 1, "model_config": model_config, "tokenizer_fingerprint": tokenizer.fingerprint},
                        trainer=current.state_dict(), sampler=train_loader.batch_sampler.state_dict())

    def best_checkpoint_callback(current: Trainer, epoch: int) -> None:
        if distributed_checkpoints:
            save_distributed_checkpoint(
                args.best_output,
                training_model,
                optimizer,
                scheduler=scheduler,
                scaler=current.scaler,
                metadata={
                    "step": current.global_step,
                    "epoch": epoch + 1,
                    "validation_loss": current.best_validation_loss,
                    "best": True,
                    "model_config": model_config,
                    "tokenizer_fingerprint": tokenizer.fingerprint,
                    "trainer": current.state_dict(),
                    "sampler": train_loader.batch_sampler.state_dict(),
                },
            )
            return
        if not distributed.is_main_process:
            return
        save_checkpoint(
            args.best_output, model, optimizer=optimizer, scheduler=scheduler, ema=ema,
            scaler=current.scaler, step=current.global_step,
            metadata={"epoch": epoch + 1, "validation_loss": current.best_validation_loss, "best": True, "model_config": model_config, "tokenizer_fingerprint": tokenizer.fingerprint},
            trainer=current.state_dict(), sampler=train_loader.batch_sampler.state_dict(),
        )

    history = trainer.fit(
        train_loader, epochs=epochs, evaluator=evaluator,
        validation_dataloader=validation_loader,
        log_every=int(config.get("log_every", 10)),
        evaluate_every=config.get("evaluate_every"),
        checkpoint_every=config.get("checkpoint_every"),
        checkpoint_callback=checkpoint_callback,
        best_checkpoint_callback=best_checkpoint_callback,
        early_stopping_patience=config.get("early_stopping_patience"),
        early_stopping_min_delta=float(config.get("early_stopping_min_delta", 0.0)),
        stop_requested=lambda: preemption.should_stop(distributed.device),
    )
    final_epoch = int(history[-1]["epoch"]) - 1 if history else trainer.current_epoch - 1
    checkpoint_callback(trainer, final_epoch)
    if distributed.is_main_process:
        print(json.dumps({
            "checkpoint": str(args.output), "best_checkpoint": str(args.best_output),
            "step": trainer.global_step, "stopped_early": trainer.stopped_early, "history": history,
        }, indent=2))
    DistributedTrainer.shutdown()


if __name__ == "__main__":
    main()
