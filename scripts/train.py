"""Train or resume the Gopi causal language model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from model.gpt import MiniGPT
from model.loss import CausalLanguageModelLoss
from optim.adamw import adamw_from_config
from optim.ema import EMA
from optim.scheduler import Scheduler
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.data import build_loader
from training.distributed import DistributedTrainer
from training.evaluator import Evaluator
from training.trainer import Trainer
from dotenv import load_dotenv

from utils.config import load_yaml
from utils.logger import configure_logging
from utils.seed import set_seed

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/latest/model.pt"))
    parser.add_argument("--best-output", type=Path, default=Path("checkpoints/best/model.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path, help="load model weights only for a new training stage")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be used together")

    configure_logging()
    model_config, config = load_yaml(args.model_config), load_yaml(args.training_config)
    if str(config.get("mixed_precision", "none")) == "fp16" and not torch.cuda.is_available():
        parser.error(
            "the selected GPU training profile requires CUDA, but PyTorch cannot access a GPU. "
            "Fix the NVIDIA driver until `nvidia-smi` works, or explicitly select "
            "--model-config configs/model.cpu.yaml --training-config configs/training.cpu.yaml"
        )
    set_seed(int(config.get("seed", 42)))
    distributed = DistributedTrainer.initialize(config.get("distributed_backend"))
    tokenizer = Tokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != int(model_config["vocab_size"]):
        parser.error("tokenizer vocabulary does not match model vocab_size")
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
        )
    training_model = DistributedTrainer.wrap(model, distributed)
    train_loader = build_loader(config["train_files"], tokenizer, config, shuffle=True, rank=distributed.rank, world_size=distributed.world_size)
    epochs = args.epochs or int(config.get("epochs", 1))
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    total_steps = max(1, (len(train_loader) * epochs + accumulation - 1) // accumulation)
    optimizer = adamw_from_config(training_model, config)
    scheduler = Scheduler.from_config(optimizer, config, total_steps=total_steps)
    ema = EMA(training_model, decay=float(config.get("ema_decay", 0.999)))
    loss_fn = CausalLanguageModelLoss.from_config(config)
    trainer = Trainer(
        training_model, optimizer, loss_fn, scheduler=scheduler, ema=ema,
        gradient_clip_norm=config.get("gradient_clip_norm", 1.0), device=distributed.device,
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(config.get("mixed_precision", "none")),
    )
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, ema=ema, scaler=trainer.scaler, map_location=distributed.device)
        trainer.global_step = state["step"]
        trainer.load_state_dict(state.get("trainer", {}))
        sampler_state = state.get("sampler")
        if sampler_state and hasattr(train_loader.batch_sampler, "load_state_dict"):
            train_loader.batch_sampler.load_state_dict(sampler_state)

    validation_loader = None
    evaluator = None
    if config.get("validation_files"):
        validation_loader = build_loader(config["validation_files"], tokenizer, config, shuffle=False, rank=distributed.rank, world_size=distributed.world_size)
        evaluator = Evaluator(training_model, loss_fn=loss_fn, device=distributed.device)

    def checkpoint_callback(current: Trainer, epoch: int) -> None:
        if not distributed.is_main_process:
            return
        save_checkpoint(args.output, model, optimizer=optimizer, scheduler=scheduler, ema=ema, scaler=current.scaler,
                        step=current.global_step, metadata={"epoch": epoch + 1, "model_config": model_config},
                        trainer=current.state_dict(), sampler=train_loader.batch_sampler.state_dict())

    def best_checkpoint_callback(current: Trainer, epoch: int) -> None:
        if not distributed.is_main_process:
            return
        save_checkpoint(
            args.best_output, model, optimizer=optimizer, scheduler=scheduler, ema=ema,
            scaler=current.scaler, step=current.global_step,
            metadata={"epoch": epoch + 1, "validation_loss": current.best_validation_loss, "best": True, "model_config": model_config},
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
