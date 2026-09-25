"""Core single-step model trainer."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping

import torch
import torch.distributed as dist
from torch import Tensor, nn

from local_dataset.sampler import CurriculumSchedule
from model.loss import (
    CausalLanguageModelLoss,
    LanguageModelLossOutput,
    MultiTokenPredictionLoss,
    MultiTokenPredictionLossOutput,
)
from optim.ema import EMA
from training.accounting import TrainingAccounting
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
        reasoning_trace_policy: str = "optional",
        mtp_loss_weight: float = 0.0,
        moe_aux_loss_weight: float = 0.0,
    ) -> None:
        self.model = model
        self.opt = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.gradient_clip_norm = gradient_clip_norm
        self.device = torch.device(device)
        self.global_step = 0
        self.micro_step = 0
        self._accumulation_tokens = 0
        self._token_normalized_window = False
        self.current_epoch = 0
        self.batch_in_epoch = 0
        self.best_validation_loss = float("inf")
        self.best_validation_domains: dict[str, float] = {}
        self.early_stopping_best_loss = float("inf")
        self.validation_metric_name: str | None = None
        self.validation_lr_last_decay_step: int | None = None
        self.epochs_without_improvement = 0
        self.stopped_early = False
        self.tokens_processed = 0
        self.training_seconds = 0.0
        self.nonfinite_updates = 0
        self.last_gradient_norm = float("nan")
        self.last_clipped_gradient_norm = float("nan")
        self.last_logit_abs_mean = float("nan")
        self.last_logit_abs_max = float("nan")
        self.last_loss = float("nan")
        self.last_parameter_norm = float("nan")
        self.last_gradient_to_parameter_ratio = float("nan")
        self.curriculum_stage = 0
        if reasoning_trace_policy not in {"optional", "assistant_only"}:
            raise ValueError("reasoning_trace_policy must be optional or assistant_only")
        self.reasoning_trace_policy = reasoning_trace_policy
        if not isinstance(mtp_loss_weight, (int, float)) or isinstance(mtp_loss_weight, bool) or not math.isfinite(float(mtp_loss_weight)) or float(mtp_loss_weight) < 0:
            raise ValueError("mtp_loss_weight must be a finite non-negative number")
        mtp_predictions = int(getattr(model, "mtp_num_predictions", 0) or 0)
        if mtp_loss_weight and mtp_predictions < 1:
            raise ValueError("mtp_loss_weight requires a model with mtp_num_predictions > 0")
        self.mtp_loss_weight = float(mtp_loss_weight)
        self.mtp_loss_fn = (
            MultiTokenPredictionLoss(mtp_predictions, weight=self.mtp_loss_weight)
            if self.mtp_loss_weight > 0 else None
        )
        self.last_mtp_loss = 0.0
        if not isinstance(moe_aux_loss_weight, (int, float)) or isinstance(moe_aux_loss_weight, bool) or not math.isfinite(float(moe_aux_loss_weight)) or float(moe_aux_loss_weight) < 0:
            raise ValueError("moe_aux_loss_weight must be a finite non-negative number")
        self.moe_aux_loss_weight = float(moe_aux_loss_weight)
        if self.moe_aux_loss_weight and not callable(getattr(model, "router_aux_loss", None)):
            raise ValueError("moe_aux_loss_weight requires a model with router_aux_loss()")
        self.last_moe_aux_loss = 0.0
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
            self._accumulation_tokens = 0
            self._token_normalized_window = False
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
            if self.reasoning_trace_policy == "assistant_only" and loss_mask is None:
                raise ValueError(
                    "reasoning_trace_policy='assistant_only' requires an explicit loss_mask "
                    "so prompt/tool tokens cannot enter the SFT objective"
                )
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
            model_kwargs = {"return_mtp_logits": True} if self.mtp_loss_fn is not None else {}
            if attention_mask is not None:
                model_output = self.model(token_ids, attention_mask=attention_mask, **model_kwargs)
            else:
                model_output = self.model(token_ids, **model_kwargs)
            mtp_logits = None
            if self.mtp_loss_fn is not None:
                if not isinstance(model_output, tuple) or len(model_output) != 2 or not isinstance(model_output[1], list):
                    raise RuntimeError("MTP-enabled model did not return (logits, mtp_logits)")
                logits, mtp_logits = model_output
            else:
                logits = model_output[0] if isinstance(model_output, tuple) else model_output
            self.last_logit_abs_mean = float(logits.detach().float().abs().mean())
            self.last_logit_abs_max = float(logits.detach().float().abs().max())
            loss_function = self.batch_loss_fn if is_batch else self.tensor_loss_fn
            # Reuse the count already computed by the standard loss. Custom
            # objectives retain their existing tensor-returning contract.
            details = (loss_function(logits, targets, loss_mask=loss_mask, return_details=True)
                       if type(loss_function) is CausalLanguageModelLoss else None)
            loss = (details.loss if isinstance(details, LanguageModelLossOutput)
                    else loss_function(logits, targets, loss_mask=loss_mask))
            if self.mtp_loss_fn is not None:
                assert mtp_logits is not None
                mtp_result = self.mtp_loss_fn(mtp_logits, targets, loss_mask=loss_mask)
                if not isinstance(mtp_result, MultiTokenPredictionLossOutput):
                    raise RuntimeError("MTP objective returned an invalid result")
                self.last_mtp_loss = float(mtp_result.loss.detach())
                loss = loss + mtp_result.loss
            else:
                self.last_mtp_loss = 0.0
            if self.moe_aux_loss_weight:
                router_loss = self.model.router_aux_loss()
                if router_loss is None:
                    raise RuntimeError("MoE auxiliary loss requested but no routed experts were active")
                weighted_router_loss = router_loss * self.moe_aux_loss_weight
                self.last_moe_aux_loss = float(weighted_router_loss.detach())
                loss = loss + weighted_router_loss
            else:
                self.last_moe_aux_loss = 0.0
        if not bool(torch.isfinite(loss.detach())):
            self.nonfinite_updates += 1
            self.opt.zero_grad(set_to_none=True)
            self.micro_step = 0
            self._accumulation_tokens = 0
            raise FloatingPointError(
                f"non-finite training loss at optimizer step {self.global_step}"
            )
        self.last_loss = float(loss.detach())
        # Mean-of-means overweights short answers. Accumulate token sums and
        # normalize once over the complete window, including a partial epoch.
        # The fixed scale keeps backward magnitudes near ordinary mean loss;
        # it cancels exactly in _optimizer_step and is identical on all ranks.
        self._token_normalized_window = (
            isinstance(details, LanguageModelLossOutput) and loss_function.reduction == "mean"
        )
        backward_loss = loss
        if self._token_normalized_window:
            self._accumulation_tokens += details.token_count
            backward_loss = loss * (details.token_count / 1024.0)
        self.scaler.scale(backward_loss / self.gradient_accumulation_steps).backward()
        self.tokens_processed += (details.token_count if isinstance(details, LanguageModelLossOutput)
                                  else self._count_target_tokens(
                                      targets, loss_mask, getattr(loss_function, "shift_labels", is_batch),
                                      getattr(loss_function, "ignore_index", -100),
                                  ))
        self.micro_step += 1
        if self.micro_step % self.gradient_accumulation_steps == 0:
            self._optimizer_step()
        self.training_seconds += time.perf_counter() - started
        return float(loss.detach().item())

    def _optimizer_step(self) -> bool:
        if self._token_normalized_window:
            count = torch.tensor(float(self._accumulation_tokens), device=self.device,
                                 dtype=torch.float32 if self.device.type == "mps" else torch.float64)
            world_size = 1
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
            total_tokens = float(count.item())
            if not total_tokens:
                # Do not apply AdamW decay, advance LR, or update EMA when a
                # truncation/mask leaves the entire window without targets.
                self.opt.zero_grad(set_to_none=True)
                return False
            # DDP/FSDP average gradients across ranks, so undo that average
            # before dividing by the global number of supervised tokens.
            correction = 1024.0 * self.gradient_accumulation_steps * world_size / total_tokens
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
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
        self.last_clipped_gradient_norm = (
            min(self.last_gradient_norm, float(self.gradient_clip_norm))
            if self.gradient_clip_norm is not None else self.last_gradient_norm
        )
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.last_parameter_norm = float(self._parameter_norm())
        self.last_gradient_to_parameter_ratio = (
            self.last_gradient_norm / self.last_parameter_norm
            if self.last_parameter_norm else float("inf")
        )
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

    def _parameter_norm(self) -> Tensor:
        """Return a scalar parameter-scale diagnostic without retaining snapshots."""
        norms = [parameter.detach().float().norm(2) for parameter in self.model.parameters()]
        if not norms:
            return torch.tensor(0.0, device=self.device)
        return torch.stack([norm.to(self.device) for norm in norms]).norm(2)

    @staticmethod
    def _count_target_tokens(
        targets: Tensor, loss_mask: Tensor | None, shift_labels: bool, ignore_index: int = -100
    ) -> int:
        selected = targets.ne(ignore_index)
        if loss_mask is not None:
            selected = selected & loss_mask.bool()
        if shift_labels and targets.ndim == 2:
            selected = selected[:, 1:]
        return int(selected.sum().item())

    @property
    def learning_rate(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_processed / self.training_seconds if self.training_seconds > 0 else 0.0

    def accounting_report(self, *, device_count: int = 1) -> dict[str, int | float]:
        """Return reproducible accounting from counters persisted in trainer state."""
        return TrainingAccounting(
            parameters=sum(parameter.numel() for parameter in self.model.parameters()),
            supervised_tokens=self.tokens_processed,
            optimizer_steps=self.global_step,
            elapsed_seconds=self.training_seconds,
            device_count=device_count,
        ).report()

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
            correction = 1.0 if self._token_normalized_window else self.gradient_accumulation_steps / remainder
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
        validation_max_batches: int | Mapping[str, int] | None = None,
        validation_progress_every: int = 0,
        validation_evaluate_at_start: bool = False,
        save_initial_best_checkpoint: bool = False,
        log_every: int = 10,
        log_interval_seconds: float | None = None,
        evaluate_every: int | None = None,
        checkpoint_every: int | None = None,
        checkpoint_callback=None,
        best_checkpoint_callback=None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        validation_lr_adaptation_enabled: bool = True,
        validation_lr_decay_factor: float | None = None,
        validation_lr_patience: int = 1,
        validation_lr_min_scale: float = 0.1,
        validation_lr_min_steps_between_decays: int = 0,
        validation_control_domain: str | None = None,
        best_checkpoint_domain_max_regression: Mapping[str, float] | None = None,
        best_checkpoint_min_generation_accuracy: float | None = None,
        validation_metric_name: str | None = None,
        validation_callback=None,
        stop_requested=None,
        curriculum_schedule: CurriculumSchedule | None = None,
    ) -> list[dict[str, object]]:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if not isinstance(validation_lr_adaptation_enabled, bool):
            raise TypeError("validation_lr_adaptation_enabled must be a boolean")
        if validation_lr_decay_factor is not None and not 0 < validation_lr_decay_factor < 1:
            raise ValueError("validation_lr_decay_factor must be between zero and one")
        if validation_lr_patience < 1:
            raise ValueError("validation_lr_patience must be positive")
        if not 0 < validation_lr_min_scale <= 1:
            raise ValueError("validation_lr_min_scale must be in (0, 1]")
        if validation_lr_min_steps_between_decays < 0:
            raise ValueError("validation_lr_min_steps_between_decays must be non-negative")
        if (
            best_checkpoint_min_generation_accuracy is not None
            and not 0 <= best_checkpoint_min_generation_accuracy <= 1
        ):
            raise ValueError("best checkpoint generation accuracy must be in [0, 1]")
        if (
            validation_lr_adaptation_enabled
            and validation_lr_decay_factor is not None
            and not callable(getattr(self.scheduler, "reduce_after_validation", None))
        ):
            raise ValueError("validation-driven LR decay requires a compatible scheduler")
        if isinstance(validation_max_batches, Mapping):
            if any(int(value) < 1 for value in validation_max_batches.values()):
                raise ValueError("validation_max_batches values must be positive")
        elif validation_max_batches is not None and int(validation_max_batches) < 1:
            raise ValueError("validation_max_batches must be positive")
        if validation_progress_every < 0:
            raise ValueError("validation_progress_every must be non-negative")
        domain_regression_limits = {
            str(name): float(value)
            for name, value in (best_checkpoint_domain_max_regression or {}).items()
        }
        if any(value < 0 for value in domain_regression_limits.values()):
            raise ValueError("best checkpoint domain regression limits must be non-negative")
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
            self.best_validation_domains = {}
            self.early_stopping_best_loss = float("inf")
            self.epochs_without_improvement = 0
        history: list[dict[str, object]] = []
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        if curriculum_schedule is not None and not isinstance(curriculum_schedule, CurriculumSchedule):
            raise TypeError("curriculum_schedule must be a CurriculumSchedule")
        # A resumable sampler reports only its remaining batches from __len__.
        # Progress and ETA need the full epoch length, independent of that
        # resume offset. Other iterable loaders retain their ordinary length.
        batches_per_epoch = int(
            getattr(batch_sampler, "total_batches", len(dataloader))
        )
        total_batches = max(1, batches_per_epoch * epochs)
        last_log_time = time.perf_counter()

        def save_timed(callback, epoch: int, kind: str) -> None:
            started = time.perf_counter()
            callback(self, epoch)
            logger.info("checkpoint kind=%s step=%d duration_seconds=%.2f",
                        kind, self.global_step, time.perf_counter() - started)

        def evaluate_validation() -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
            logger.info("validation_started step=%d", self.global_step)
            started = time.perf_counter()
            metrics, domains = evaluate_validation_metrics()
            logger.info("validation_timing step=%d duration_seconds=%.2f",
                        self.global_step, time.perf_counter() - started)
            return metrics, domains

        def evaluate_validation_metrics() -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
            def batch_limit(name: str | None = None) -> int | None:
                if isinstance(validation_max_batches, Mapping):
                    return (
                        int(validation_max_batches[name])
                        if name in validation_max_batches else None
                    )
                return int(validation_max_batches) if validation_max_batches is not None else None

            def evaluate_loader(loader, name: str | None = None):
                limit = batch_limit(name)
                if limit is None and not validation_progress_every:
                    return evaluator.evaluate(loader)
                return evaluator.evaluate(
                    loader, max_batches=limit,
                    progress_every=validation_progress_every,
                    label=name or "validation",
                )

            if isinstance(validation_dataloader, Mapping):
                if validation_weights is None:
                    raise ValueError("validation_weights are required for domain validation loaders")
                domains = {
                    str(name): evaluate_loader(loader, str(name))
                    for name, loader in validation_dataloader.items()
                }
                return aggregate_domain_metrics(domains, validation_weights), domains
            return evaluate_loader(validation_dataloader), {}

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
                "validation epoch=%d step=%d loss=%.6f cross_entropy=%.6f perplexity=%.4f "
                "tokens=%d batches=%d metric=%s",
                epoch + 1, self.global_step, float(metrics["loss"]),
                float(metrics.get("cross_entropy", float("nan"))),
                float(metrics.get("perplexity", float("nan"))),
                int(metrics.get("tokens", 0)), int(metrics.get("batches", 0)),
                self.validation_metric_name or "validation_loss",
            )

        def update_from_validation(validation_loss: float) -> None:
            if validation_loss < self.early_stopping_best_loss - early_stopping_min_delta:
                self.early_stopping_best_loss = validation_loss
                self.epochs_without_improvement = 0
                return
            self.epochs_without_improvement += 1
            if (
                validation_lr_adaptation_enabled
                and validation_lr_decay_factor is not None
                and self.epochs_without_improvement % validation_lr_patience == 0
                and (
                    self.validation_lr_last_decay_step is None
                    or self.global_step - self.validation_lr_last_decay_step
                    >= validation_lr_min_steps_between_decays
                )
            ):
                previous, current = self.scheduler.reduce_after_validation(
                    validation_lr_decay_factor, min_scale=validation_lr_min_scale
                )
                if current < previous:
                    self.validation_lr_last_decay_step = self.global_step
                    logger.info(
                        "validation_lr_decay step=%d loss=%.6f bad_checks=%d "
                        "scale=%.6f->%.6f lr=%.8g",
                        self.global_step, validation_loss, self.epochs_without_improvement,
                        previous, current, self.learning_rate,
                    )

        def validation_control_loss(metrics, domains) -> float:
            if validation_control_domain is None:
                return float(metrics["loss"])
            if validation_control_domain not in domains:
                raise ValueError(
                    f"validation control domain {validation_control_domain!r} is unavailable"
                )
            return float(domains[validation_control_domain]["loss"])

        def checkpoint_passes_domain_gates(domains) -> bool:
            for name, tolerance in domain_regression_limits.items():
                if name not in domains:
                    raise ValueError(f"best checkpoint gate domain {name!r} is unavailable")
                baseline = self.best_validation_domains.get(name)
                if baseline is not None and float(domains[name]["loss"]) > baseline * (1.0 + tolerance):
                    logger.info(
                        "best_checkpoint_rejected step=%d domain=%s loss=%.6f limit=%.6f",
                        self.global_step, name, float(domains[name]["loss"]),
                        baseline * (1.0 + tolerance),
                    )
                    return False
            return True

        def checkpoint_passes_generation_gate(generation_metrics) -> bool:
            threshold = best_checkpoint_min_generation_accuracy
            if generation_metrics and generation_metrics.get("retention_passed") is False:
                logger.info(
                    "best_checkpoint_rejected step=%d reason=retention_regression",
                    self.global_step,
                )
                return False
            if threshold is None:
                return True
            accuracy = generation_metrics.get("accuracy") if generation_metrics else None
            if accuracy is None or float(accuracy) < threshold:
                logger.info(
                    "best_checkpoint_rejected step=%d generation_accuracy=%s minimum=%.4f",
                    self.global_step, "missing" if accuracy is None else f"{float(accuracy):.4f}",
                    threshold,
                )
                return False
            return True

        def record_best_domains(domains) -> None:
            self.best_validation_domains = {
                str(name): float(values["loss"]) for name, values in domains.items()
            }

        if (
            validation_evaluate_at_start
            and evaluator is not None
            and validation_dataloader is not None
            and (
                not self.best_validation_domains
                or (save_initial_best_checkpoint and math.isinf(self.best_validation_loss))
            )
        ):
            metrics, domains = evaluate_validation()
            log_validation(self.current_epoch - 1, metrics, domains)
            if not domains and domain_regression_limits:
                raise ValueError(
                    "step-zero domain retention gates require domain validation loaders"
                )
            record_best_domains(domains)
            initial_validation_loss = float(metrics["loss"])
            self.early_stopping_best_loss = validation_control_loss(metrics, domains)
            self.epochs_without_improvement = 0
            if save_initial_best_checkpoint and math.isinf(self.best_validation_loss):
                self.best_validation_loss = initial_validation_loss
                logger.info(
                    "initial_best_validation step=%d loss=%.6f metric=%s",
                    self.global_step, initial_validation_loss,
                    self.validation_metric_name or "validation_loss",
                )
                if best_checkpoint_callback:
                    save_timed(
                        best_checkpoint_callback, self.current_epoch - 1, "initial_best"
                    )
            if self.device.type == "cuda":
                # Evaluation and checkpointing create allocation shapes that
                # differ from backward. Return their unused cached blocks so
                # the first training batch can reserve contiguous workspace
                # on memory-constrained GPUs.
                allocated = torch.cuda.memory_allocated(self.device) / 1024**2
                reserved = torch.cuda.memory_reserved(self.device) / 1024**2
                torch.cuda.empty_cache()
                logger.info(
                    "released_initial_validation_cuda_cache "
                    "allocated_mb=%.1f reserved_mb=%.1f->%.1f",
                    allocated, reserved,
                    torch.cuda.memory_reserved(self.device) / 1024**2,
                )
            history.append({
                "epoch": self.current_epoch,
                "step": self.global_step,
                "initial_validation": True,
                **metrics,
                **({"domains": domains} if domains else {}),
            })
            last_log_time = time.perf_counter()

        for epoch in range(self.current_epoch, epochs):
            last_log_time = time.perf_counter()
            last_validation_step = None
            if curriculum_schedule is not None:
                stage_index, stage = curriculum_schedule.stage_for_epoch(epoch)
                if batch_sampler is None or not hasattr(batch_sampler, "set_sampling_group_weights"):
                    raise ValueError("curriculum scheduling requires a grouped resumable batch sampler")
                batch_sampler.set_sampling_group_weights(stage.weights)
                self.curriculum_stage = stage_index
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            if hasattr(batch_sampler, "set_start_batch"):
                batch_sampler.set_start_batch(self.batch_in_epoch if epoch == self.current_epoch else 0)
            running_loss = 0.0
            running_batches = 0
            window_loss = 0.0
            window_batches = 0
            window_training_seconds = 0.0
            resume_offset = self.batch_in_epoch if epoch == self.current_epoch else 0
            batch_index = resume_offset
            for batch_index, batch in enumerate(dataloader, resume_offset + 1):
                previous_step = self.global_step
                step_started = time.perf_counter()
                loss = self.train_step(batch)
                window_training_seconds += time.perf_counter() - step_started
                self.batch_in_epoch = batch_index
                running_loss += loss
                running_batches += 1
                window_loss += loss
                window_batches += 1
                optimizer_stepped = self.global_step != previous_step
                now = time.perf_counter()
                log_due = (
                    log_interval_seconds is not None
                    and now - last_log_time >= log_interval_seconds
                ) or (
                    log_interval_seconds is None
                    and log_every
                    and self.global_step % log_every == 0
                )
                if optimizer_stepped and log_due:
                    current_loss = window_loss / max(window_batches, 1)
                    avg_loss = running_loss / max(running_batches, 1)
                    completed_batches = min(
                        total_batches, epoch * batches_per_epoch + batch_index
                    )
                    progress = completed_batches / total_batches
                    epoch_progress = min(1.0, batch_index / max(1, batches_per_epoch))
                    elapsed_seconds = self.training_seconds
                    eta_seconds = (
                        elapsed_seconds * (1.0 - progress) / progress
                        if progress > 0 else float("inf")
                    )
                    seconds_per_batch = window_training_seconds / max(window_batches, 1)
                    epoch_batches_left = max(0, batches_per_epoch - batch_index)

                    def event_eta(interval, enabled=True, *, epoch_end=False, next_log=False):
                        if not enabled:
                            return "disabled"
                        remaining = None
                        if interval:
                            steps = (-self.global_step) % interval
                            if next_log and steps == 0:
                                steps = interval
                            remaining = steps * self.gradient_accumulation_steps
                        if epoch_end:
                            remaining = min(remaining, epoch_batches_left) if remaining is not None else epoch_batches_left
                        # Epoch-end gradient flushing can change the step schedule.
                        if remaining is None or remaining > epoch_batches_left:
                            return "after_epoch"
                        return f"{remaining * seconds_per_batch:.1f}"

                    next_log_eta = event_eta(log_every, next_log=True)
                    next_checkpoint_eta = event_eta(checkpoint_every, bool(checkpoint_every and checkpoint_callback))
                    next_validation_eta = event_eta(evaluate_every, bool(evaluator and validation_dataloader), epoch_end=True)
                    allocated_mb, reserved_mb, total_mb = self.gpu_memory_mb
                    logger.info(
                        "epoch=%d step=%d loss=%.6f lr=%.8g grad_norm=%.4f "
                        "clipped_grad_norm=%.4f tokens=%d "
                        "tokens_per_second=%.1f progress=%.2f%% epoch_progress=%.2f%% elapsed_seconds=%.0f "
                        "eta_seconds=%.0f best_validation_loss=%.6f peak_memory_mb=%.1f "
                        "gpu_memory_mb=%.1f/%.1f/%.1f nonfinite_updates=%d "
                        "log_interval_seconds=%.2f seconds_per_step=%.3f "
                        "next_log_eta_seconds=%s next_checkpoint_eta_seconds=%s "
                        f"next_validation_eta_seconds=%s mtp_loss={self.last_mtp_loss:.6f} "
                        f"moe_aux_loss={self.last_moe_aux_loss:.6f} (avg=%.6f)",
                        epoch + 1, self.global_step, current_loss, self.learning_rate,
                        self.last_gradient_norm, self.last_clipped_gradient_norm,
                        self.tokens_processed, self.tokens_per_second,
                        progress * 100.0, epoch_progress * 100.0, elapsed_seconds, eta_seconds,
                        self.best_validation_loss, self.peak_memory_mb,
                        allocated_mb, reserved_mb, total_mb,
                        self.nonfinite_updates, now - last_log_time,
                        seconds_per_batch * self.gradient_accumulation_steps,
                        next_log_eta, next_checkpoint_eta, next_validation_eta, avg_loss,
                    )
                    last_log_time = now
                    window_training_seconds = 0.0
                    window_loss = 0.0
                    window_batches = 0
                if optimizer_stepped and evaluate_every and evaluator and validation_dataloader and self.global_step % evaluate_every == 0:
                    # Persist the exact training position before a potentially
                    # long validation pass. A second save below records the
                    # resulting best-loss, early-stop, and adaptive-LR state.
                    if checkpoint_every and checkpoint_callback and self.global_step % checkpoint_every == 0:
                        save_timed(checkpoint_callback, epoch, "latest_pre_validation")
                    metrics, domains = evaluate_validation()
                    log_validation(epoch, metrics, domains)
                    history.append({
                        "epoch": epoch + 1, "step": self.global_step, **metrics,
                        **({"domains": domains} if domains else {}),
                    })
                    callback_metrics = None
                    if validation_callback:
                        callback_metrics = validation_callback(self, epoch, metrics, domains)
                        if callback_metrics:
                            history[-1]["generation_evaluation"] = callback_metrics
                    last_validation_step = self.global_step
                    validation_loss = float(metrics["loss"])
                    update_from_validation(validation_control_loss(metrics, domains))
                    # The loss-selected checkpoint and the generation-selected
                    # checkpoint are separate artifacts. Generation retention
                    # is enforced while writing best-generation.pt; it must not
                    # veto an improved best.pt validation checkpoint.
                    if (validation_loss < self.best_validation_loss
                            and checkpoint_passes_domain_gates(domains)):
                        previous_best = self.best_validation_loss
                        self.best_validation_loss = validation_loss
                        record_best_domains(domains)
                        logger.info(
                            "new_best_validation step=%d previous_loss=%.6f loss=%.6f metric=%s",
                            self.global_step, previous_best, validation_loss,
                            self.validation_metric_name or "validation_loss",
                        )
                        if best_checkpoint_callback:
                            save_timed(best_checkpoint_callback, epoch, "best")
                    if early_stopping_patience is not None and self.epochs_without_improvement >= early_stopping_patience:
                        self.stopped_early = True
                        logger.info(
                            "early stopping at step=%d after %d validation checks without improvement",
                            self.global_step, self.epochs_without_improvement,
                        )
                        if checkpoint_callback:
                            save_timed(checkpoint_callback, epoch, "latest")
                        # Keep the batch offset for resume and avoid evaluating
                        # these same weights again at epoch end.
                        return history
                # When validation and checkpoint intervals coincide, persist
                # the newly updated best/early-stopping state in latest.pt.
                # Saving first would resume with the stale pre-validation
                # baseline (often infinity for a new training stage).
                if optimizer_stepped and checkpoint_every and checkpoint_callback and self.global_step % checkpoint_every == 0:
                    save_timed(checkpoint_callback, epoch, "latest")
                if optimizer_stepped and (
                    (evaluate_every and evaluator and validation_dataloader and self.global_step % evaluate_every == 0)
                    or (checkpoint_every and checkpoint_callback and self.global_step % checkpoint_every == 0)
                ):
                    last_log_time = time.perf_counter()
                if optimizer_stepped and stop_requested and stop_requested():
                    if checkpoint_callback:
                        save_timed(checkpoint_callback, epoch, "latest")
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
                "mtp_loss": self.last_mtp_loss,
                "moe_aux_loss": self.last_moe_aux_loss,
                "clipped_gradient_norm": self.last_clipped_gradient_norm,
                "loss": self.last_loss,
                "logit_abs_mean": self.last_logit_abs_mean,
                "logit_abs_max": self.last_logit_abs_max,
                "parameter_norm": self.last_parameter_norm,
                "gradient_to_parameter_ratio": self.last_gradient_to_parameter_ratio,
                "curriculum_stage": self.curriculum_stage,
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
                callback_metrics = None
                if validation_callback:
                    callback_metrics = validation_callback(self, epoch, metrics, domains)
                    if callback_metrics:
                        epoch_record["generation_evaluation"] = callback_metrics
                validation_loss = float(epoch_record["loss"])
                if last_validation_step != self.global_step:
                    update_from_validation(validation_control_loss(metrics, domains))
                if (validation_loss < self.best_validation_loss
                        and checkpoint_passes_domain_gates(domains)):
                    previous_best = self.best_validation_loss
                    self.best_validation_loss = validation_loss
                    record_best_domains(domains)
                    logger.info(
                        "new_best_validation step=%d previous_loss=%.6f loss=%.6f metric=%s",
                        self.global_step, previous_best, validation_loss,
                        self.validation_metric_name or "validation_loss",
                    )
                    if best_checkpoint_callback:
                        save_timed(best_checkpoint_callback, epoch, "best")
            history.append(epoch_record)
            if self.stopped_early:
                break
            if early_stopping_patience is not None and self.epochs_without_improvement >= early_stopping_patience:
                self.stopped_early = True
                logger.info(
                    "early stopping at epoch=%d after %d validation checks without improvement",
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
            "best_validation_domains": self.best_validation_domains,
            "early_stopping_best_loss": self.early_stopping_best_loss,
            "validation_metric_name": self.validation_metric_name,
            "validation_lr_last_decay_step": self.validation_lr_last_decay_step,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.stopped_early,
            "tokens_processed": self.tokens_processed,
            "training_seconds": self.training_seconds,
            "nonfinite_updates": self.nonfinite_updates,
            "last_gradient_norm": self.last_gradient_norm,
            "last_clipped_gradient_norm": self.last_clipped_gradient_norm,
            "last_logit_abs_mean": self.last_logit_abs_mean,
            "last_logit_abs_max": self.last_logit_abs_max,
            "last_loss": self.last_loss,
            "last_mtp_loss": self.last_mtp_loss,
            "last_moe_aux_loss": self.last_moe_aux_loss,
            "last_parameter_norm": self.last_parameter_norm,
            "last_gradient_to_parameter_ratio": self.last_gradient_to_parameter_ratio,
            "curriculum_stage": self.curriculum_stage,
        }

    def load_state_dict(self, state: Mapping[str, int | float | bool | str | None]) -> None:
        self.global_step = int(state.get("global_step", self.global_step))
        self.micro_step = int(state.get("micro_step", 0))
        self.current_epoch = int(state.get("current_epoch", 0))
        self.batch_in_epoch = int(state.get("batch_in_epoch", 0))
        self.best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        saved_domains = state.get("best_validation_domains", {})
        self.best_validation_domains = {
            str(name): float(value) for name, value in saved_domains.items()
        } if isinstance(saved_domains, Mapping) else {}
        self.early_stopping_best_loss = float(
            state.get("early_stopping_best_loss", self.best_validation_loss)
        )
        metric_name = state.get("validation_metric_name")
        self.validation_metric_name = str(metric_name) if metric_name is not None else None
        last_decay_step = state.get("validation_lr_last_decay_step")
        self.validation_lr_last_decay_step = (
            int(last_decay_step) if last_decay_step is not None else None
        )
        self.epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        self.stopped_early = bool(state.get("stopped_early", False))
        self.tokens_processed = int(state.get("tokens_processed", 0))
        self.training_seconds = float(state.get("training_seconds", 0.0))
        self.nonfinite_updates = int(state.get("nonfinite_updates", 0))
        self.last_gradient_norm = float(state.get("last_gradient_norm", float("nan")))
        self.last_clipped_gradient_norm = float(
            state.get("last_clipped_gradient_norm", self.last_gradient_norm)
        )
        self.last_logit_abs_mean = float(state.get("last_logit_abs_mean", float("nan")))
        self.last_logit_abs_max = float(state.get("last_logit_abs_max", float("nan")))
        self.last_loss = float(state.get("last_loss", float("nan")))
        self.last_mtp_loss = float(state.get("last_mtp_loss", 0.0))
        self.last_moe_aux_loss = float(state.get("last_moe_aux_loss", 0.0))
        self.last_parameter_norm = float(state.get("last_parameter_norm", float("nan")))
        self.last_gradient_to_parameter_ratio = float(
            state.get("last_gradient_to_parameter_ratio", float("nan"))
        )
        self.curriculum_stage = int(state.get("curriculum_stage", 0))

    def promotion_metrics(self) -> dict[str, float]:
        """Expose deterministic training-side metrics for multi-axis checkpoint selection."""
        return {
            "validation_loss": float(self.best_validation_loss),
            "tokens_processed": float(self.tokens_processed),
            "tokens_per_second": float(self.tokens_per_second),
            "training_seconds": float(self.training_seconds),
        }
