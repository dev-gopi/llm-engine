"""Token-weighted language-model evaluation loop."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
import torch.distributed as dist
from torch import Tensor, nn

from model.loss import CausalLanguageModelLoss, LanguageModelLossOutput


class Evaluator:
    def __init__(self, model: nn.Module, *, loss_fn: CausalLanguageModelLoss | None = None, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.loss_fn = loss_fn or CausalLanguageModelLoss(shift_labels=True, reduction="mean")

    @torch.inference_mode()
    def evaluate(self, dataloader: Iterable[Mapping[str, Tensor]], *, max_batches: int | None = None) -> dict[str, float | int]:
        was_training = self.model.training
        self.model.eval()
        loss_sum = 0.0
        cross_entropy_sum = 0.0
        z_loss_sum = 0.0
        token_count = 0
        batch_count = 0
        try:
            for batch_count, batch in enumerate(dataloader, 1):
                if max_batches is not None and batch_count > max_batches:
                    batch_count -= 1
                    break
                inputs = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                loss_mask = batch.get("loss_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                if loss_mask is not None:
                    loss_mask = loss_mask.to(self.device)
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
