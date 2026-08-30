"""Token-weighted language-model evaluation loop."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
import torch.distributed as dist
from torch import Tensor, nn

from model.loss import CausalLanguageModelLoss, LanguageModelLossOutput
from utils.logger import get_logger


logger = get_logger(__name__)


def aggregate_domain_metrics(
    domains: Mapping[str, Mapping[str, float | int]],
    weights: Mapping[str, float],
) -> dict[str, float | int]:
    """Aggregate fixed domain evaluations with explicit capability weights."""
    if not domains:
        raise ValueError("domain evaluation results cannot be empty")
    missing = set(domains) - set(weights)
    if missing:
        raise ValueError(f"validation weights missing domains: {sorted(missing)}")
    selected = {name: float(weights[name]) for name in domains}
    if any(weight < 0 for weight in selected.values()) or not any(selected.values()):
        raise ValueError("validation weights must contain a positive non-negative weight")
    total_weight = sum(selected.values())
    aggregate: dict[str, float | int] = {}
    for key in ("loss", "cross_entropy", "z_loss"):
        aggregate[key] = sum(
            float(domains[name][key]) * weight for name, weight in selected.items()
        ) / total_weight
    aggregate["perplexity"] = math.exp(min(float(aggregate["cross_entropy"]), 80.0))
    aggregate["tokens"] = sum(int(metrics["tokens"]) for metrics in domains.values())
    aggregate["batches"] = sum(int(metrics["batches"]) for metrics in domains.values())
    return aggregate


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        *,
        loss_fn: CausalLanguageModelLoss | None = None,
        device: str | torch.device = "cpu",
        mixed_precision: str = "none",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.loss_fn = loss_fn or CausalLanguageModelLoss(shift_labels=True, reduction="mean")
        if mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if mixed_precision == "fp16" and self.device.type != "cuda":
            logger.warning(
                "FP16 evaluation requires CUDA; falling back to full precision on %s",
                self.device.type,
            )
            mixed_precision = "none"
        if mixed_precision == "bf16" and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 evaluation requires a BF16-capable CUDA device")
        if mixed_precision != "none" and self.device.type not in {"cpu", "cuda"}:
            logger.warning(
                "mixed-precision evaluation is not enabled for %s; falling back to full precision",
                self.device.type,
            )
            mixed_precision = "none"
        self.mixed_precision = mixed_precision
        self.autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16

    @torch.inference_mode()
    def evaluate(self, dataloader: Iterable[Mapping[str, Tensor]], *, max_batches: int | None = None) -> dict[str, float | int]:
        was_training = self.model.training
        self.model.eval()
        loss_sum = 0.0
        cross_entropy_sum = 0.0
        z_loss_sum = 0.0
        token_count = 0
        batch_count = 0
        non_blocking = self.device.type == "cuda"
        try:
            for batch_count, batch in enumerate(dataloader, 1):
                if max_batches is not None and batch_count > max_batches:
                    batch_count -= 1
                    break
                inputs = batch["input_ids"].to(self.device, non_blocking=non_blocking)
                labels = batch["labels"].to(self.device, non_blocking=non_blocking)
                attention_mask = batch.get("attention_mask")
                loss_mask = batch.get("loss_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, non_blocking=non_blocking)
                if loss_mask is not None:
                    loss_mask = loss_mask.to(self.device, non_blocking=non_blocking)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.mixed_precision != "none",
                ):
                    output = self.model(inputs, attention_mask=attention_mask) if attention_mask is not None else self.model(inputs)
                    logits = output[0] if isinstance(output, tuple) else output
                    details = self.loss_fn(logits, labels, loss_mask=loss_mask, return_details=True)
                if not isinstance(details, LanguageModelLossOutput):
                    raise RuntimeError("loss function did not return detailed metrics")
                loss_sum += float(details.loss) * details.token_count
                cross_entropy_sum += float(details.cross_entropy) * details.token_count
                z_loss_sum += float(details.z_loss) * details.token_count
                token_count += details.token_count
        finally:
            self.model.train(was_training)
        if dist.is_available() and dist.is_initialized():
            totals = torch.tensor(
                [loss_sum, cross_entropy_sum, z_loss_sum, token_count, batch_count],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            loss_sum, cross_entropy_sum, z_loss_sum = map(float, totals[:3])
            token_count, batch_count = int(totals[3]), int(totals[4])
        if token_count == 0:
            raise ValueError("evaluation produced no valid target tokens")
        loss = loss_sum / token_count
        cross_entropy = cross_entropy_sum / token_count
        return {
            "loss": loss,
            "cross_entropy": cross_entropy,
            "z_loss": z_loss_sum / token_count,
            "perplexity": math.exp(min(cross_entropy, 80.0)),
            "tokens": token_count,
            "batches": batch_count,
        }
