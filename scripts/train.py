"""Train or resume the Gopi causal language model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

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
from utils.config import load_yaml
from utils.logger import configure_logging
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/latest/model.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    configure_logging()
    model_config, config = load_yaml(args.model_config), load_yaml(args.training_config)
    set_seed(int(config.get("seed", 42)))
    distributed = DistributedTrainer.initialize(config.get("distributed_backend"))
    tokenizer = Tokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != int(model_config["vocab_size"]):
        parser.error("tokenizer vocabulary does not match model vocab_size")
    model = MiniGPT.from_config(model_config, device=distributed.device)
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
                        step=current.global_step, metadata={"epoch": epoch + 1},
                        trainer=current.state_dict(), sampler=train_loader.batch_sampler.state_dict())

    history = trainer.fit(
        train_loader, epochs=epochs, evaluator=evaluator,
        validation_dataloader=validation_loader,
        log_every=int(config.get("log_every", 10)),
        evaluate_every=config.get("evaluate_every"),
        checkpoint_every=config.get("checkpoint_every"),
        checkpoint_callback=checkpoint_callback,
    )
    checkpoint_callback(trainer, epochs - 1)
    if distributed.is_main_process:
        print(json.dumps({"checkpoint": str(args.output), "step": trainer.global_step, "history": history}, indent=2))
    DistributedTrainer.shutdown()


if __name__ == "__main__":
    main()
