"""Production pipeline-parallel partitioning/training for :class:`model.gpt.MiniGPT`.

The runtime partitions decoder blocks across the PP axis, keeps TP/EP/CP/SP
modifications already applied to those blocks, and executes a non-interleaved
1F1B schedule with multiple microbatches.  Data-parallel wrapping can be applied
independently to each local stage.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from model.loss import CausalLanguageModelLoss, LanguageModelLossOutput
from training.context_parallel import shard_batch_sequence
from utils.logger import get_logger

logger = get_logger(__name__)


def balanced_layer_range(num_layers: int, stages: int, stage: int) -> tuple[int, int]:
    if num_layers < 1 or stages < 1 or not 0 <= stage < stages:
        raise ValueError("invalid pipeline partition dimensions")
    if stages > num_layers:
        raise ValueError("pipeline degree cannot exceed the number of transformer layers")
    base, extra = divmod(num_layers, stages)
    start = stage * base + min(stage, extra)
    stop = start + base + (1 if stage < extra else 0)
    return start, stop


class MiniGPTPipelinePartition(nn.Module):
    """One MiniGPT pipeline stage owning only its local trainable modules."""

    def __init__(self, model: nn.Module, *, stage: int, stages: int) -> None:
        super().__init__()
        self.stage = int(stage)
        self.stages = int(stages)
        self.first = self.stage == 0
        self.last = self.stage == self.stages - 1
        self.dim = int(model.dim)
        self.vocab_size = int(model.vocab_size)
        self.max_positions = int(model.max_positions)
        self.position_type = str(model.position_type)
        self.gradient_checkpointing = bool(getattr(model, "gradient_checkpointing", False))
        self.context_parallel_group = getattr(model, "context_parallel_group", None)
        self.context_parallel_rank = int(getattr(model, "context_parallel_rank", 0) or 0)
        self.context_parallel_size = int(getattr(model, "context_parallel_size", 1) or 1)
        self.sequence_parallel_group = getattr(model, "sequence_parallel_group", None)
        self.logit_softcap = getattr(model, "logit_softcap", None)
        self.mtp_num_predictions = int(getattr(model, "mtp_num_predictions", 0) or 0)

        start, stop = balanced_layer_range(len(model.blocks), self.stages, self.stage)
        self.layer_start, self.layer_stop = start, stop
        self.blocks = nn.ModuleList(list(model.blocks[start:stop]))

        if self.first:
            self.tok = model.tok
            self.pos = model.pos
            self.embedding_dropout = model.embedding_dropout
        else:
            self.tok = None
            self.pos = None
            self.embedding_dropout = None

        # RoPE has no trainable ownership requirement, but each PP stage needs
        # the cache/function to project its local attention blocks.
        self.rotary_emb = model.rotary_emb

        if self.last:
            self.norm = model.norm
            self.head = model.head
            self.mtp_heads = model.mtp_heads
        else:
            self.norm = None
            self.head = None
            self.mtp_heads = nn.ModuleList()

    def _position_ids(self, token_ids: Tensor, attention_mask: Tensor | None) -> Tensor | None:
        seq_len = token_ids.shape[1]
        offset = self.context_parallel_rank * seq_len if self.context_parallel_size > 1 else 0
        if attention_mask is None:
            return None
        return attention_mask.long().cumsum(dim=1).sub(1).clamp_min(0) + offset

    def _rotary(self, hidden: Tensor, position_ids: Tensor | None):
        if self.position_type != "rotary" or self.rotary_emb is None:
            return None
        seq_len = hidden.shape[1]
        offset = self.context_parallel_rank * seq_len if self.context_parallel_size > 1 else 0
        rope_length = offset + seq_len
        if position_ids is not None and position_ids.numel():
            rope_length = max(rope_length, int(position_ids.max().detach().cpu()) + 1)
        rotary = self.rotary_emb(hidden, seq_len=rope_length)
        if position_ids is None:
            rotary = (rotary[0][:, :, offset : offset + seq_len, :], rotary[1][:, :, offset : offset + seq_len, :])
        return rotary

    def forward(self, value: Tensor, *, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Run this stage.

        ``value`` is token IDs on stage zero and hidden activations otherwise.
        ``token_ids`` is supplied on every rank only to derive positions/masks.
        """
        if self.first:
            if value.dtype != torch.long:
                value = value.long()
            position_ids = self._position_ids(token_ids, attention_mask)
            seq_len = token_ids.shape[1]
            offset = self.context_parallel_rank * seq_len if self.context_parallel_size > 1 else 0
            if self.position_type == "learned" and self.pos is not None:
                positions = self.pos(token_ids, position_ids=position_ids, position_offset=offset, attention_mask=attention_mask)
                hidden = self.embedding_dropout(self.tok(token_ids) + positions)
            elif self.position_type == "sinusoidal" and self.pos is not None:
                positions = self.pos(token_ids, position_ids=position_ids, position_offset=offset)
                hidden = self.embedding_dropout(self.tok(token_ids) + positions)
            else:
                hidden = self.embedding_dropout(self.tok(token_ids))
        else:
            hidden = value
            position_ids = self._position_ids(token_ids, attention_mask)

        rotary = self._rotary(hidden, position_ids)
        for block in self.blocks:
            hidden = block(
                hidden,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary,
                position_ids=position_ids,
            )
            if isinstance(hidden, tuple):
                raise RuntimeError("pipeline training does not use KV-cache outputs")

        if not self.last:
            return hidden
        assert self.norm is not None and self.head is not None
        logits = self.head(self.norm(hidden))
        if self.logit_softcap is not None:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits

    def router_aux_loss(self) -> Tensor | None:
        losses = []
        for block in self.blocks:
            ffn = getattr(block, "ffn", None)
            value = getattr(ffn, "last_router_aux_loss", None)
            if value is not None:
                losses.append(value)
        return torch.stack(losses).mean() if losses else None

    def partition_metadata(self) -> dict[str, int]:
        return {"stage": self.stage, "stages": self.stages, "layer_start": self.layer_start, "layer_stop": self.layer_stop}


@dataclass
class _MicrobatchState:
    input_activation: Tensor | None
    output_activation: Tensor
    batch: dict[str, Tensor]
    loss: Tensor | None = None


class PipelineTrainer:
    """MiniGPT PP trainer using a multi-microbatch non-interleaved 1F1B schedule."""

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        loss_fn: CausalLanguageModelLoss,
        *,
        mesh,
        scheduler=None,
        device="cpu",
        microbatches: int = 1,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float | None = 1.0,
        tracker=None,
        moe_aux_loss_weight: float = 0.0,
        deepspeed_engine: bool = False,
    ) -> None:
        if mesh is None or mesh.degree("pipeline") <= 1:
            raise ValueError("PipelineTrainer requires pipeline degree > 1")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("pipeline training requires torch.distributed")
        if microbatches < 1:
            raise ValueError("pipeline microbatches must be positive")
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.model = model
        self.opt = optimizer
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.mesh = mesh
        self.group = mesh.group("pipeline")
        self.members = mesh.members("pipeline")
        self.stage = mesh.local_rank("pipeline")
        self.stages = mesh.degree("pipeline")
        self.microbatches = int(microbatches)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.gradient_clip_norm = gradient_clip_norm
        self.tracker = tracker
        self.moe_aux_loss_weight = float(moe_aux_loss_weight)
        self.deepspeed_engine = bool(deepspeed_engine)
        self.global_step = 0
        self.micro_step = 0
        self.current_epoch = 0
        self.batch_in_epoch = 0
        self.best_validation_loss = float("inf")
        self.best_validation_domains: dict[str, float] = {}
        self.early_stopping_best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.stopped_early = False
        self.tokens_processed = 0
        self.training_seconds = 0.0
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self._pending_sends: list[tuple[object, Tensor]] = []

    @property
    def partition(self) -> nn.Module:
        return getattr(self.model, "module", self.model)

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "current_epoch": self.current_epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "best_validation_loss": self.best_validation_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "tokens_processed": self.tokens_processed,
            "training_seconds": self.training_seconds,
        }

    def load_state_dict(self, state):
        for key in self.state_dict():
            if key in state:
                setattr(self, key, state[key])

    def _split_batch(self, batch: Mapping[str, Tensor]) -> list[dict[str, Tensor]]:
        size = int(batch["input_ids"].shape[0])
        count = min(self.microbatches, size)
        if count < 1:
            raise ValueError("empty pipeline batch")
        boundaries = torch.tensor_split(torch.arange(size), count)
        result = []
        for idx in boundaries:
            result.append({key: value.index_select(0, idx) for key, value in batch.items() if isinstance(value, Tensor)})
        return result

    def _local_batch(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        values = {k: v.to(self.device, non_blocking=self.device.type == "cuda") for k, v in batch.items() if isinstance(v, Tensor)}
        cp = self.mesh.degree("context")
        if cp > 1:
            values = shard_batch_sequence(values, rank=self.mesh.local_rank("context"), world_size=cp, keys=("input_ids", "attention_mask", "labels", "loss_mask"))
        return values

    def _shape_for(self, mb: dict[str, Tensor]) -> tuple[int, int, int]:
        return (int(mb["input_ids"].shape[0]), int(mb["input_ids"].shape[1]), int(self.partition.dim))

    def _isend(self, tensor: Tensor, dst: int, tag: int):
        buffer = tensor.detach().contiguous()
        request = dist.isend(buffer, dst=dst, group=self.group, tag=tag)
        self._pending_sends.append((request, buffer))

    def _forward_microbatch(self, mb_index: int, mb: dict[str, Tensor], *, train: bool) -> _MicrobatchState:
        token_ids = mb["input_ids"]
        attention = mb.get("attention_mask")
        if self.stage == 0:
            input_activation = None
            value = token_ids
        else:
            value = torch.empty(self._shape_for(mb), device=self.device, dtype=next(self.partition.parameters()).dtype)
            dist.recv(value, src=self.members[self.stage - 1], group=self.group, tag=mb_index)
            if train:
                value.requires_grad_(True)
            input_activation = value
        output = self.model(value, token_ids=token_ids, attention_mask=attention)
        if self.stage < self.stages - 1:
            self._isend(output, self.members[self.stage + 1], mb_index)
            return _MicrobatchState(input_activation, output, mb)
        labels = mb["labels"]
        loss_mask = mb.get("loss_mask")
        details = self.loss_fn(output, labels, loss_mask=loss_mask, return_details=True)
        loss = details.loss if isinstance(details, LanguageModelLossOutput) else details
        if self.moe_aux_loss_weight:
            aux = self.partition.router_aux_loss()
            if aux is not None:
                loss = loss + self.moe_aux_loss_weight * aux
        return _MicrobatchState(input_activation, output, mb, loss)

    def _backward_microbatch(
        self, mb_index: int, state: _MicrobatchState, microbatch_count: int
    ):
        if self.stage == self.stages - 1:
            assert state.loss is not None
            if self.deepspeed_engine:
                # Average pipeline microbatches here; DeepSpeed applies its own
                # gradient-accumulation scaling exactly once inside backward.
                self.model.backward(state.loss / microbatch_count)
            else:
                (
                    state.loss
                    / (microbatch_count * self.gradient_accumulation_steps)
                ).backward()
        else:
            grad = torch.empty_like(state.output_activation)
            dist.recv(
                grad,
                src=self.members[self.stage + 1],
                group=self.group,
                tag=100000 + mb_index,
            )
            if self.deepspeed_engine:
                # The gradient received from the next stage is already scaled by
                # DeepSpeed on the last stage. Disable a second GAS scaling when
                # entering this stage's ZeRO hooks through a scalar surrogate.
                surrogate = (state.output_activation * grad).sum()
                try:
                    self.model.backward(surrogate, scale_wrt_gas=False)
                except TypeError:
                    # Compatibility fallback for engine-like test doubles. Real
                    # supported DeepSpeed releases expose ``scale_wrt_gas``.
                    surrogate.backward()
            else:
                state.output_activation.backward(grad)
        if self.stage > 0:
            if state.input_activation is None or state.input_activation.grad is None:
                raise RuntimeError("pipeline stage did not produce an input gradient")
            self._isend(state.input_activation.grad, self.members[self.stage - 1], 100000 + mb_index)

    def _wait_sends(self):
        for request, _buffer in self._pending_sends:
            request.wait()
        self._pending_sends.clear()

    def train_step(self, batch: Mapping[str, Tensor]) -> float:
        started = time.perf_counter()
        self.model.train()
        if (
            not self.deepspeed_engine
            and self.micro_step % self.gradient_accumulation_steps == 0
        ):
            self.opt.zero_grad(set_to_none=True)
        local = self._local_batch(batch)
        microbatches = self._split_batch(local)
        n = len(microbatches)
        warmup = min(self.stages - self.stage - 1, n)
        remaining = n - warmup
        states: dict[int, _MicrobatchState] = {}
        detached_losses: list[float] = []

        for mb in range(warmup):
            states[mb] = self._forward_microbatch(mb, microbatches[mb], train=True)
            if states[mb].loss is not None:
                detached_losses.append(float(states[mb].loss.detach()))
        for i in range(remaining):
            fwd = warmup + i
            states[fwd] = self._forward_microbatch(fwd, microbatches[fwd], train=True)
            if states[fwd].loss is not None:
                detached_losses.append(float(states[fwd].loss.detach()))
            self._backward_microbatch(i, states.pop(i), n)
        for i in range(warmup):
            mb = remaining + i
            self._backward_microbatch(mb, states.pop(mb), n)
        self._wait_sends()

        self.micro_step += 1
        should_step = self.micro_step % self.gradient_accumulation_steps == 0
        grad_norm = torch.tensor(float("nan"), device=self.device)
        if self.deepspeed_engine:
            # Query the boundary before ``step``; DeepSpeed advances its internal
            # micro-step counter during the step call.
            boundary = getattr(self.model, "is_gradient_accumulation_boundary", None)
            should_step = bool(boundary()) if callable(boundary) else should_step
            self.model.step()
            if should_step:
                self.global_step += 1
                self.micro_step = 0
        elif should_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm
                if self.gradient_clip_norm is not None
                else float("inf"),
            )
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                self.opt.zero_grad(set_to_none=True)
                raise FloatingPointError("non-finite pipeline gradient norm")
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
        self.batch_in_epoch += 1
        local_tokens = int(local.get("loss_mask", local.get("attention_mask", torch.ones_like(local["input_ids"]))).sum().item())
        # Count tokens once per PP group; DP reductions/reporters can aggregate later.
        self.tokens_processed += local_tokens
        self.training_seconds += time.perf_counter() - started

        loss_value = (sum(detached_losses) / len(detached_losses)) if detached_losses else 0.0
        loss_tensor = torch.tensor(loss_value, device=self.device, dtype=torch.float64)
        dist.broadcast(loss_tensor, src=self.members[-1], group=self.group)
        if self.tracker is not None and self.stage == 0:
            metrics = {"train/pipeline_loss": float(loss_tensor), "train/tokens": float(local_tokens)}
            if should_step:
                metrics["train/grad_norm"] = float(grad_norm)
            self.tracker.log(metrics, step=self.global_step)
        return float(loss_tensor)

    def flush_accumulation(self) -> None:
        remainder = self.micro_step % self.gradient_accumulation_steps
        if not remainder:
            return
        if self.deepspeed_engine:
            # Native PP only permits ZeRO-0/1, where local parameter gradients
            # remain available. Rescale the partial window and force a real
            # DeepSpeed accumulation boundary so the final batches are not lost.
            setter = getattr(self.model, "set_gradient_accumulation_boundary", None)
            if callable(setter):
                scale = self.gradient_accumulation_steps / remainder
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
                setter(True)
                try:
                    self.model.step()
                finally:
                    setter(False)
                self.global_step += 1
                self.micro_step = 0
                return
            logger.warning(
                "DeepSpeed engine cannot force an accumulation boundary; "
                "discarding incomplete window (%d/%d)",
                remainder,
                self.gradient_accumulation_steps,
            )
            self.model.zero_grad()
            self.micro_step = 0
            return
        # Gradients were normalized for a full accumulation window. Correct the
        # partial final window before the optimizer update.
        scale = self.gradient_accumulation_steps / remainder
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip_norm if self.gradient_clip_norm is not None else float("inf"),
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            self.opt.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite pipeline gradient norm")
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        self.opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate(self, loader, *, max_batches: int | None = None):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches = 0
        for batch in loader:
            local = self._local_batch(batch)
            for mb_index, mb in enumerate(self._split_batch(local)):
                state = self._forward_microbatch(mb_index, mb, train=False)
                if self.stage == self.stages - 1:
                    assert state.loss is not None
                    mask = mb.get("loss_mask", mb.get("attention_mask"))
                    tokens = int(mask.sum().item()) if mask is not None else int(mb["labels"].numel())
                    total_loss += float(state.loss) * tokens
                    total_tokens += tokens
                self._wait_sends()
            batches += 1
            if max_batches is not None and batches >= int(max_batches):
                break
        packed = torch.tensor([total_loss, float(total_tokens), float(batches)], device=self.device, dtype=torch.float64)
        dist.broadcast(packed, src=self.members[-1], group=self.group)
        if packed[1].item() <= 0:
            raise ValueError("pipeline validation loader is empty")
        loss = float(packed[0] / packed[1])
        return {"loss": loss, "cross_entropy": loss, "perplexity": math.exp(min(loss, 80.0)), "tokens": int(packed[1]), "batches": int(packed[2])}

    def fit(
        self,
        dataloader,
        *,
        epochs: int,
        evaluator=None,
        validation_dataloader=None,
        validation_weights=None,
        validation_max_batches=None,
        validation_progress_every=0,
        validation_evaluate_at_start=False,
        save_initial_best_checkpoint=False,
        log_every=10,
        log_interval_seconds=None,
        evaluate_every=None,
        checkpoint_every=None,
        checkpoint_callback=None,
        best_checkpoint_callback=None,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        validation_lr_adaptation_enabled=True,
        validation_lr_decay_factor=None,
        validation_lr_patience=1,
        validation_lr_min_scale=0.1,
        validation_lr_min_steps_between_decays=0,
        validation_control_domain=None,
        best_checkpoint_domain_max_regression=None,
        best_checkpoint_min_generation_accuracy=None,
        validation_metric_name=None,
        validation_callback=None,
        stop_requested=None,
        curriculum_schedule=None,
    ):
        if isinstance(validation_dataloader, Mapping):
            raise ValueError("pipeline trainer currently requires one validation loader, not domain-mapped loaders")
        history = []
        if validation_evaluate_at_start and validation_dataloader is not None:
            metrics = self.evaluate(validation_dataloader, max_batches=validation_max_batches if isinstance(validation_max_batches, int) else None)
            self.best_validation_loss = float(metrics["loss"])
            if save_initial_best_checkpoint and best_checkpoint_callback:
                best_checkpoint_callback(self, self.current_epoch - 1)
        for epoch in range(self.current_epoch, int(epochs)):
            self.current_epoch = epoch
            sampler = getattr(dataloader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self.batch_in_epoch = 0
            for batch in dataloader:
                if stop_requested is not None and stop_requested():
                    self.stopped_early = True
                    break
                self.train_step(batch)
                if checkpoint_every and self.global_step % int(checkpoint_every) == 0 and checkpoint_callback:
                    checkpoint_callback(self, epoch)
                if evaluate_every and validation_dataloader is not None and self.global_step % int(evaluate_every) == 0:
                    metrics = self.evaluate(validation_dataloader, max_batches=validation_max_batches if isinstance(validation_max_batches, int) else None)
                    if validation_callback:
                        validation_callback(self, epoch, metrics, {})
            self.flush_accumulation()
            row = {"epoch": epoch + 1, "step": self.global_step}
            if validation_dataloader is not None:
                metrics = self.evaluate(validation_dataloader, max_batches=validation_max_batches if isinstance(validation_max_batches, int) else None)
                row.update({f"validation_{k}": v for k, v in metrics.items()})
                if validation_callback:
                    extra = validation_callback(self, epoch, metrics, {})
                    if isinstance(extra, Mapping):
                        row.update(extra)
                val = float(metrics["loss"])
                if val < self.best_validation_loss - float(early_stopping_min_delta):
                    self.best_validation_loss = val
                    self.epochs_without_improvement = 0
                    if best_checkpoint_callback:
                        best_checkpoint_callback(self, epoch)
                else:
                    self.epochs_without_improvement += 1
                    if early_stopping_patience is not None and self.epochs_without_improvement >= int(early_stopping_patience):
                        self.stopped_early = True
            if checkpoint_callback:
                checkpoint_callback(self, epoch)
            history.append(row)
            if self.stopped_early:
                break
            self.current_epoch = epoch + 1
        return history


def build_pipeline_partition(model: nn.Module, mesh) -> MiniGPTPipelinePartition:
    if mesh is None or mesh.degree("pipeline") <= 1:
        raise ValueError("pipeline partition requires pipeline degree > 1")
    return MiniGPTPipelinePartition(model, stage=mesh.local_rank("pipeline"), stages=mesh.degree("pipeline"))


__all__ = ["balanced_layer_range", "MiniGPTPipelinePartition", "PipelineTrainer", "build_pipeline_partition"]
