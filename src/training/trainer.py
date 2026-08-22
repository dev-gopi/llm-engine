"""Core single-step model trainer."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from model.loss import CausalLanguageModelLoss
from optim.ema import EMA
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
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if mixed_precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires CUDA")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16")
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
        self.model.train()
        if self.micro_step % self.gradient_accumulation_steps == 0:
            self.opt.zero_grad(set_to_none=True)
        attention_mask = None
        loss_mask = None
        is_batch = isinstance(inputs, Mapping)
        if is_batch:
            batch = inputs
            token_ids = batch["input_ids"].to(self.device)
            targets = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")
            loss_mask = batch.get("loss_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device)
        else:
            token_ids = inputs.to(self.device)
            if targets is None:
                raise ValueError("targets are required when inputs is a tensor")
            targets = targets.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.mixed_precision != "none"):
            logits = self.model(token_ids, attention_mask=attention_mask) if attention_mask is not None else self.model(token_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss_function = self.batch_loss_fn if is_batch else self.tensor_loss_fn
            loss = loss_function(logits, targets, loss_mask=loss_mask)
        self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
        self.micro_step += 1
        if self.micro_step % self.gradient_accumulation_steps == 0:
            self._optimizer_step()
        return float(loss.detach().item())

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.opt)
        if self.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.global_step += 1

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
        log_every: int = 10,
        evaluate_every: int | None = None,
        checkpoint_every: int | None = None,
        checkpoint_callback=None,
    ) -> list[dict[str, float | int]]:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        history: list[dict[str, float | int]] = []
        for epoch in range(self.current_epoch, epochs):
            batch_sampler = getattr(dataloader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            if hasattr(batch_sampler, "set_start_batch"):
                batch_sampler.set_start_batch(self.batch_in_epoch if epoch == self.current_epoch else 0)
            running_loss = 0.0
            resume_offset = self.batch_in_epoch if epoch == self.current_epoch else 0
            batch_index = resume_offset
            for batch_index, batch in enumerate(dataloader, resume_offset + 1):
                previous_step = self.global_step
                loss = self.train_step(batch)
                self.batch_in_epoch = batch_index
                running_loss += loss
                optimizer_stepped = self.global_step != previous_step
                if optimizer_stepped and log_every and self.global_step % log_every == 0:
                    logger.info("epoch=%d step=%d loss=%.6f", epoch + 1, self.global_step, running_loss / batch_index)
                if optimizer_stepped and checkpoint_every and checkpoint_callback and self.global_step % checkpoint_every == 0:
                    checkpoint_callback(self, epoch)
                if optimizer_stepped and evaluate_every and evaluator and validation_dataloader and self.global_step % evaluate_every == 0:
                    metrics = evaluator.evaluate(validation_dataloader)
                    history.append({"epoch": epoch + 1, "step": self.global_step, **metrics})
            self.flush_gradients()
            self.current_epoch = epoch + 1
            self.batch_in_epoch = 0
            if hasattr(batch_sampler, "set_start_batch"):
                batch_sampler.set_start_batch(0)
            epoch_record: dict[str, float | int] = {"epoch": epoch + 1, "step": self.global_step, "train_loss": running_loss / max(batch_index, 1)}
            if evaluator and validation_dataloader:
                epoch_record.update(evaluator.evaluate(validation_dataloader))
            history.append(epoch_record)
        return history

    def state_dict(self) -> dict[str, int]:
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "current_epoch": self.current_epoch,
            "batch_in_epoch": self.batch_in_epoch,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self.global_step = int(state.get("global_step", self.global_step))
        self.micro_step = int(state.get("micro_step", 0))
        self.current_epoch = int(state.get("current_epoch", 0))
        self.batch_in_epoch = int(state.get("batch_in_epoch", 0))
