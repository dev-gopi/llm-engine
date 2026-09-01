"""Core single-step model trainer."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from model.loss import CausalLanguageModelLoss
from optim.ema import EMA
from training.evaluator import aggregate_domain_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        loss_fn: CausalLanguageModelLoss | None = None,
        *,
        scheduler=None,
        ema: EMA | None = None,
        gradient_clip_norm: float | None = 1.0,
        device: str | torch.device = "cpu",
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "none",
        grad_scaler_initial_scale: float = 65536.0,
        grad_scaler_growth_interval: int = 2000,
    ) -> None:
        self.model = model
        self.opt = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.gradient_clip_norm = gradient_clip_norm
        self.device = torch.device(device)
        self.global_step = 0
        self.micro_step = 0
        self.current_epoch = 0
        self.batch_in_epoch = 0
        self.best_validation_loss = float("inf")
        self.early_stopping_best_loss = float("inf")
        self.validation_metric_name: str | None = None
        self.epochs_without_improvement = 0
        self.stopped_early = False
        self.tokens_processed = 0
        self.training_seconds = 0.0
        self.nonfinite_updates = 0
        self.last_gradient_norm = float("nan")
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if mixed_precision == "fp16" and self.device.type != "cuda":
            logger.warning("fp16 mixed precision requested but CUDA device is not available. Falling back to mixed_precision='none'.")
            mixed_precision = "none"
        if grad_scaler_initial_scale <= 0:
            raise ValueError("grad_scaler_initial_scale must be positive")
        if grad_scaler_growth_interval < 1:
            raise ValueError("grad_scaler_growth_interval must be positive")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=grad_scaler_initial_scale,
            growth_interval=grad_scaler_growth_interval,
            enabled=mixed_precision == "fp16" and self.device.type == "cuda",
        )
        self.model.to(self.device)
        # Existing trainer callers provide explicit next-token targets, so this
        # integration does not shift them a second time.
        self.loss_fn = loss_fn or CausalLanguageModelLoss(shift_labels=False)
        self.tensor_loss_fn = self.loss_fn
        self.batch_loss_fn = loss_fn or CausalLanguageModelLoss(shift_labels=True)

    def train_step(
        self,
        inputs: Tensor | Mapping[str, Tensor],
        targets: Tensor | None = None,
    ) -> float:
        started = time.perf_counter()
        self.model.train()
        if self.micro_step % self.gradient_accumulation_steps == 0:
            self.opt.zero_grad(set_to_none=True)
        attention_mask = None
        loss_mask = None
        is_batch = isinstance(inputs, Mapping)
        non_blocking = self.device.type == "cuda"
        if is_batch:
            batch = inputs
            token_ids = batch["input_ids"].to(self.device, non_blocking=non_blocking)
            targets = batch["labels"].to(self.device, non_blocking=non_blocking)
            attention_mask = batch.get("attention_mask")
            loss_mask = batch.get("loss_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device, non_blocking=non_blocking)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device, non_blocking=non_blocking)
        else:
            token_ids = inputs.to(self.device, non_blocking=non_blocking)
            if targets is None:
                raise ValueError("targets are required when inputs is a tensor")
            targets = targets.to(self.device, non_blocking=non_blocking)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.mixed_precision != "none"):
            logits = self.model(token_ids, attention_mask=attention_mask) if attention_mask is not None else self.model(token_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss_function = self.batch_loss_fn if is_batch else self.tensor_loss_fn
            loss = loss_function(logits, targets, loss_mask=loss_mask)
        if not bool(torch.isfinite(loss.detach())):
            self.nonfinite_updates += 1
            self.opt.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"non-finite training loss at optimizer step {self.global_step}"
            )
        self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
        self.tokens_processed += self._count_target_tokens(targets, loss_mask, is_batch)
        self.micro_step += 1
        if self.micro_step % self.gradient_accumulation_steps == 0:
            self._optimizer_step()
        self.training_seconds += time.perf_counter() - started
        return float(loss.detach().item())

    def _optimizer_step(self) -> bool:
        self.scaler.unscale_(self.opt)
        if self.gradient_clip_norm is not None:
            # FSDP must aggregate sharded gradient norms collectively.
            clip = getattr(self.model, "clip_grad_norm_", None)
            if callable(clip):
                gradient_norm = clip(self.gradient_clip_norm)
            else:
                gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        else:
            gradient_norm = self._gradient_norm()
        self.last_gradient_norm = float(gradient_norm)
        if not math.isfinite(self.last_gradient_norm):
            self.nonfinite_updates += 1
            self.opt.zero_grad(set_to_none=True)
            self.scaler.update()
            logger.warning(
                "skipping non-finite gradients at optimizer step=%d gradient_norm=%s "
                "loss_scale=%.1f; the update was discarded and the FP16 loss scale was reduced",
                self.global_step,
                self.last_gradient_norm,
                self.scaler.get_scale(),
            )
            return False
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.global_step += 1
        return True

    def _gradient_norm(self) -> Tensor:
        norms = [
            parameter.grad.detach().float().norm(2)
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=self.device)
        return torch.stack([norm.to(self.device) for norm in norms]).norm(2)

    @staticmethod
    def _count_target_tokens(targets: Tensor, loss_mask: Tensor | None, is_batch: bool) -> int:
        if loss_mask is not None:
            selected = loss_mask[:, 1:] if is_batch and loss_mask.ndim == 2 else loss_mask
            return int(selected.sum().item())
        return int(targets[:, 1:].numel() if is_batch and targets.ndim == 2 else targets.numel())

    @property
    def learning_rate(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_processed / self.training_seconds if self.training_seconds > 0 else 0.0

    @property
    def peak_memory_mb(self) -> float:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

    @property
    def gpu_memory_mb(self) -> tuple[float, float, float]:
        """Return current allocated, reserved, and total CUDA memory in MiB."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return (0.0, 0.0, 0.0)
        divisor = 1024 * 1024
        return (
            torch.cuda.memory_allocated(self.device) / divisor,
            torch.cuda.memory_reserved(self.device) / divisor,
            torch.cuda.get_device_properties(self.device).total_memory / divisor,
        )

    def flush_gradients(self) -> None:
        remainder = self.micro_step % self.gradient_accumulation_steps
        if remainder:
            correction = self.gradient_accumulation_steps / remainder
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            self._optimizer_step()
            self.opt.zero_grad(set_to_none=True)
            self.micro_step = 0

    def fit(
        self,
        dataloader: Iterable[Mapping[str, Tensor]],
        *,
        epochs: int,
        evaluator=None,
        validation_dataloader=None,
        validation_weights: Mapping[str, float] | None = None,
        log_every: int = 10,
        evaluate_every: int | None = None,
        checkpoint_every: int | None = None,
        checkpoint_callback=None,
        best_checkpoint_callback=None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        validation_metric_name: str | None = None,
        stop_requested=None,
    ) -> list[dict[str, object]]:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if (
            validation_metric_name is not None
            and validation_metric_name != self.validation_metric_name
        ):
            logger.info(
                "validation metric changed from %r to %r; resetting best-loss baseline",
                self.validation_metric_name, validation_metric_name,
            )
            self.validation_metric_name = validation_metric_name
            self.best_validation_loss = float("inf")
            self.early_stopping_best_loss = float("inf")
            self.epochs_without_improvement = 0
        history: list[dict[str, object]] = []
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        # A resumable sampler reports only its remaining batches from __len__.
        # Progress and ETA need the full epoch length, independent of that
        # resume offset. Other iterable loaders retain their ordinary length.
        batches_per_epoch = int(
            getattr(batch_sampler, "total_batches", len(dataloader))
        )
        total_batches = max(1, batches_per_epoch * epochs)

        def evaluate_validation() -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
            if isinstance(validation_dataloader, Mapping):
                if validation_weights is None:
                    raise ValueError("validation_weights are required for domain validation loaders")
                domains = {
                    str(name): evaluator.evaluate(loader)
                    for name, loader in validation_dataloader.items()
                }
                return aggregate_domain_metrics(domains, validation_weights), domains
            return evaluator.evaluate(validation_dataloader), {}

        def log_validation(epoch: int, metrics: Mapping[str, float | int], domains) -> None:
            for domain, domain_metrics in domains.items():
                logger.info(
                    "validation_domain=%s epoch=%d step=%d loss=%.6f cross_entropy=%.6f "
                    "perplexity=%.4f tokens=%d batches=%d",
                    domain, epoch + 1, self.global_step,
                    float(domain_metrics["loss"]), float(domain_metrics["cross_entropy"]),
                    float(domain_metrics["perplexity"]), int(domain_metrics["tokens"]),
                    int(domain_metrics["batches"]),
                )
            logger.info(
                "validation epoch=%d step=%d loss=%.6f cross_entropy=%.6f perplexity=%.4f tokens=%d batches=%d",
                epoch + 1, self.global_step, float(metrics["loss"]),
                float(metrics.get("cross_entropy", float("nan"))),
                float(metrics.get("perplexity", float("nan"))),
                int(metrics.get("tokens", 0)), int(metrics.get("batches", 0)),
            )

        for epoch in range(self.current_epoch, epochs):
            improved_during_epoch = False
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            if hasattr(batch_sampler, "set_start_batch"):
                batch_sampler.set_start_batch(self.batch_in_epoch if epoch == self.current_epoch else 0)
            running_loss = 0.0
            running_batches = 0
            window_loss = 0.0
            window_batches = 0
            resume_offset = self.batch_in_epoch if epoch == self.current_epoch else 0
            batch_index = resume_offset
            for batch_index, batch in enumerate(dataloader, resume_offset + 1):
                previous_step = self.global_step
                loss = self.train_step(batch)
                self.batch_in_epoch = batch_index
                running_loss += loss
                running_batches += 1
                window_loss += loss
                window_batches += 1
                optimizer_stepped = self.global_step != previous_step
                if optimizer_stepped and log_every and self.global_step % log_every == 0:
                    current_loss = window_loss / max(window_batches, 1)
                    avg_loss = running_loss / max(running_batches, 1)
                    completed_batches = min(
                        total_batches, epoch * batches_per_epoch + batch_index
                    )
                    progress = completed_batches / total_batches
                    elapsed_seconds = self.training_seconds
                    eta_seconds = (
                        elapsed_seconds * (1.0 - progress) / progress
                        if progress > 0 else float("inf")
                    )
                    allocated_mb, reserved_mb, total_mb = self.gpu_memory_mb
                    logger.info(
                        "epoch=%d step=%d loss=%.6f lr=%.8g grad_norm=%.4f tokens=%d "
                        "tokens_per_second=%.1f progress=%.2f%% elapsed_seconds=%.0f "
                        "eta_seconds=%.0f best_validation_loss=%.6f peak_memory_mb=%.1f "
                        "gpu_memory_mb=%.1f/%.1f/%.1f nonfinite_updates=%d (avg=%.6f)",
                        epoch + 1, self.global_step, current_loss, self.learning_rate,
                        self.last_gradient_norm, self.tokens_processed, self.tokens_per_second,
                        progress * 100.0, elapsed_seconds, eta_seconds,
                        self.best_validation_loss, self.peak_memory_mb,
                        allocated_mb, reserved_mb, total_mb,
                        self.nonfinite_updates, avg_loss,
                    )
                    window_loss = 0.0
                    window_batches = 0
                if optimizer_stepped and evaluate_every and evaluator and validation_dataloader and self.global_step % evaluate_every == 0:
                    metrics, domains = evaluate_validation()
                    log_validation(epoch, metrics, domains)
                    history.append({
                        "epoch": epoch + 1, "step": self.global_step, **metrics,
                        **({"domains": domains} if domains else {}),
                    })
                    validation_loss = float(metrics["loss"])
                    if validation_loss < self.early_stopping_best_loss - early_stopping_min_delta:
                        self.early_stopping_best_loss = validation_loss
                        self.epochs_without_improvement = 0
                        improved_during_epoch = True
                    if validation_loss < self.best_validation_loss:
                        previous_best = self.best_validation_loss
                        self.best_validation_loss = validation_loss
                        logger.info(
                            "new_best_validation step=%d previous_loss=%.6f loss=%.6f metric=%s",
                            self.global_step, previous_best, validation_loss,
                            self.validation_metric_name or "validation_loss",
                        )
                        if best_checkpoint_callback:
                            best_checkpoint_callback(self, epoch)
                # When validation and checkpoint intervals coincide, persist
                # the newly updated best/early-stopping state in latest.pt.
                # Saving first would resume with the stale pre-validation
                # baseline (often infinity for a new training stage).
                if optimizer_stepped and checkpoint_every and checkpoint_callback and self.global_step % checkpoint_every == 0:
                    checkpoint_callback(self, epoch)
                if optimizer_stepped and stop_requested and stop_requested():
                    if checkpoint_callback:
                        checkpoint_callback(self, epoch)
                    self.stopped_early = True
                    logger.warning("coordinated preemption requested at step=%d", self.global_step)
                    break
            self.flush_gradients()
            self.current_epoch = epoch + 1
            self.batch_in_epoch = 0
            if hasattr(batch_sampler, "set_start_batch"):
                batch_sampler.set_start_batch(0)
            epoch_record: dict[str, object] = {
                "epoch": epoch + 1,
                "step": self.global_step,
                "train_loss": running_loss / max(running_batches, 1),
                "learning_rate": self.learning_rate,
                "gradient_norm": self.last_gradient_norm,
                "tokens_processed": self.tokens_processed,
                "tokens_per_second": self.tokens_per_second,
                "peak_memory_mb": self.peak_memory_mb,
                "nonfinite_updates": self.nonfinite_updates,
            }
            if evaluator and validation_dataloader:
                metrics, domains = evaluate_validation()
                epoch_record.update(metrics)
                if domains:
                    epoch_record["domains"] = domains
                log_validation(epoch, metrics, domains)
                validation_loss = float(epoch_record["loss"])
                if validation_loss < self.early_stopping_best_loss - early_stopping_min_delta:
                    self.early_stopping_best_loss = validation_loss
                    self.epochs_without_improvement = 0
                    improved_during_epoch = True
                if validation_loss < self.best_validation_loss:
                    previous_best = self.best_validation_loss
                    self.best_validation_loss = validation_loss
                    logger.info(
                        "new_best_validation step=%d previous_loss=%.6f loss=%.6f metric=%s",
                        self.global_step, previous_best, validation_loss,
                        self.validation_metric_name or "validation_loss",
                    )
                    if best_checkpoint_callback:
                        best_checkpoint_callback(self, epoch)
                if not improved_during_epoch:
                    self.epochs_without_improvement += 1
            history.append(epoch_record)
            if self.stopped_early:
                break
            if early_stopping_patience is not None and self.epochs_without_improvement >= early_stopping_patience:
                self.stopped_early = True
                logger.info(
                    "early stopping at epoch=%d after %d epochs without validation improvement",
                    epoch + 1, self.epochs_without_improvement,
                )
                break
        return history

    def state_dict(self) -> dict[str, int | float | bool | str | None]:
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "current_epoch": self.current_epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "best_validation_loss": self.best_validation_loss,
            "early_stopping_best_loss": self.early_stopping_best_loss,
            "validation_metric_name": self.validation_metric_name,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.stopped_early,
            "tokens_processed": self.tokens_processed,
            "training_seconds": self.training_seconds,
            "nonfinite_updates": self.nonfinite_updates,
            "last_gradient_norm": self.last_gradient_norm,
        }

    def load_state_dict(self, state: Mapping[str, int | float | bool | str | None]) -> None:
        self.global_step = int(state.get("global_step", self.global_step))
        self.micro_step = int(state.get("micro_step", 0))
        self.current_epoch = int(state.get("current_epoch", 0))
        self.batch_in_epoch = int(state.get("batch_in_epoch", 0))
        self.best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        self.early_stopping_best_loss = float(
            state.get("early_stopping_best_loss", self.best_validation_loss)
        )
        metric_name = state.get("validation_metric_name")
        self.validation_metric_name = str(metric_name) if metric_name is not None else None
        self.epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        self.stopped_early = bool(state.get("stopped_early", False))
        self.tokens_processed = int(state.get("tokens_processed", 0))
        self.training_seconds = float(state.get("training_seconds", 0.0))
        self.nonfinite_updates = int(state.get("nonfinite_updates", 0))
        self.last_gradient_norm = float(state.get("last_gradient_norm", float("nan")))
