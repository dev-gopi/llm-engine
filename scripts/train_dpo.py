"""Train a policy with Direct Preference Optimization on chosen/rejected pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from datasets.governance import enforce_dataset_governance
from model.gpt import MiniGPT
from optim.adamw import adamw_from_config
from optim.scheduler import Scheduler
from post_training.dpo import DPOTrainer
from post_training.preference_data import build_preference_loader
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device
from utils.logger import configure_logging, get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.v2.gpu.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/dpo.v2.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2"))
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/v2-dpo/latest.pt"))
    parser.add_argument("--best-output", type=Path, default=Path("checkpoints/v2-dpo/best.pt"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be used together")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        parser.error("DPO CLI currently supports single-device training only")

    configure_logging()
    model_config = load_yaml(args.model_config)
    config = load_yaml(args.training_config)
    set_seed(int(config.get("seed", 42)))
    paths = [*config.get("train_files", []), *config.get("validation_files", [])]
    for finding in enforce_dataset_governance(paths, config.get("dataset_governance")):
        logger.warning("dataset governance [%s]: %s", finding.code, finding.message)
    device = resolve_device(args.device)
    mixed_precision = str(config.get("mixed_precision", "none"))
    if mixed_precision == "fp16" and device.type != "cuda":
        parser.error("the selected DPO profile requires CUDA fp16; use configs/dpo.v2.cpu.yaml")
    if mixed_precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("the selected DPO profile requires BF16, but this GPU does not support it")
    tokenizer = Tokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != int(model_config["vocab_size"]):
        parser.error("tokenizer vocabulary does not match model vocab_size")
    if int(config["max_sequence_length"]) > int(model_config["max_position"]):
        parser.error("DPO max_sequence_length exceeds the model context length")

    policy = MiniGPT.from_config(model_config, device=device)
    reference = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(
        args.reference_checkpoint, reference, map_location=device, use_ema=True,
        restore_rng=False, expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    if not args.resume:
        load_checkpoint(
            args.init_from or args.reference_checkpoint, policy, map_location=device,
            use_ema=True, restore_rng=False,
            expected_tokenizer_fingerprint=tokenizer.fingerprint,
        )
    train_loader = build_preference_loader(
        config["train_files"], tokenizer, max_length=int(config["max_sequence_length"]),
        batch_size=int(config["batch_size"]), shuffle=True, seed=int(config.get("seed", 42)),
        num_workers=int(config.get("num_workers", 0)),
    )
    validation_loader = build_preference_loader(
        config["validation_files"], tokenizer, max_length=int(config["max_sequence_length"]),
        batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    ) if config.get("validation_files") else None
    epochs = args.epochs or int(config.get("epochs", 1))
    optimizer = adamw_from_config(policy, config)
    scheduler = Scheduler.from_config(optimizer, config, total_steps=max(1, len(train_loader) * epochs))
    trainer = DPOTrainer(
        policy, reference, optimizer, beta=float(config.get("beta", 0.1)),
        label_smoothing=float(config.get("label_smoothing", 0.0)), scheduler=scheduler,
        gradient_clip_norm=config.get("gradient_clip_norm", 1.0), mixed_precision=mixed_precision,
    )
    if args.resume:
        state = load_checkpoint(
            args.resume, policy, optimizer=optimizer, scheduler=scheduler,
            scaler=trainer.scaler, map_location=device,
            expected_tokenizer_fingerprint=tokenizer.fingerprint,
        )
        trainer.load_state_dict(state.get("trainer", {}))

    def save(path: Path, current: DPOTrainer, epoch: int, *, best: bool = False) -> None:
        save_checkpoint(
            path, policy, optimizer=optimizer, scheduler=scheduler, scaler=current.scaler,
            step=current.global_step, trainer=current.state_dict(), metadata={
                "epoch": epoch + 1, "best": best, "model_config": model_config,
                "reference_checkpoint": str(args.reference_checkpoint), "training_type": "dpo",
                "tokenizer_fingerprint": tokenizer.fingerprint,
            },
        )

    history = trainer.fit(
        train_loader, epochs=epochs, validation_loader=validation_loader,
        checkpoint_callback=lambda current, epoch: save(args.output, current, epoch),
        best_checkpoint_callback=lambda current, epoch: save(args.best_output, current, epoch, best=True),
        early_stopping_patience=config.get("early_stopping_patience"),
        log_every=int(config.get("log_every", 10)),
    )
    print(json.dumps({
        "checkpoint": str(args.output), "best_checkpoint": str(args.best_output),
        "step": trainer.global_step, "stopped_early": trainer.stopped_early, "history": history,
    }, indent=2))


if __name__ == "__main__":
    main()
