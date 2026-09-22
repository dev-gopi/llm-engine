"""Position-wise feed-forward network for Transformer blocks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

STANDARD_ACTIVATIONS = frozenset({"gelu", "gelu_tanh", "relu", "silu"})
GATED_ACTIVATIONS = frozenset({"swiglu", "geglu"})
SUPPORTED_ACTIVATIONS = STANDARD_ACTIVATIONS | GATED_ACTIVATIONS


class FeedForward(nn.Module):
    """Apply an independent nonlinear transformation at every token position.

    Standard activations use ``Linear → activation → Linear``. SwiGLU and GEGLU
    use one fused input projection for the gate and value branches, followed by
    an output projection. The fused layout is efficient on accelerators and
    keeps checkpoint structure straightforward.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        *,
        expansion_factor: float = 4.0,
        multiple_of: int = 1,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        initializer_range: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            dim,
            hidden_dim,
            expansion_factor,
            multiple_of,
            activation,
            dropout,
            initializer_range,
        )
        requested_hidden_dim = hidden_dim or math.ceil(dim * expansion_factor)
        self.dim = dim
        self.hidden_dim = math.ceil(requested_hidden_dim / multiple_of) * multiple_of
        self.activation_name = activation
        self.dropout_probability = float(dropout)
        self.initializer_range = float(initializer_range)
        self.is_gated = activation in GATED_ACTIVATIONS

        factory_kwargs = {"device": device, "dtype": dtype}
        input_features = 2 * self.hidden_dim if self.is_gated else self.hidden_dim
        self.in_proj = nn.Linear(dim, input_features, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.tensor_parallel_group = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.initializer_range)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        self._validate_hidden_states(hidden_states)
        projected = self.in_proj(hidden_states)
        if self.is_gated:
            gate, value = projected.chunk(2, dim=-1)
            hidden = self._activate_gate(gate) * value
        else:
            hidden = self._activate(projected)
        output = self.dropout(self.out_proj(hidden))
        if self.tensor_parallel_group is not None:
            torch.distributed.all_reduce(output, group=self.tensor_parallel_group)
        return output

    def _activate(self, hidden_states: Tensor) -> Tensor:
        if self.activation_name == "gelu":
            return F.gelu(hidden_states)
        if self.activation_name == "gelu_tanh":
            return F.gelu(hidden_states, approximate="tanh")
        if self.activation_name == "relu":
            return F.relu(hidden_states)
        if self.activation_name == "silu":
            return F.silu(hidden_states)
        raise RuntimeError(f"unsupported standard activation: {self.activation_name}")

    def _activate_gate(self, gate: Tensor) -> Tensor:
        if self.activation_name == "swiglu":
            return F.silu(gate)
        if self.activation_name == "geglu":
            return F.gelu(gate)
        raise RuntimeError(f"unsupported gated activation: {self.activation_name}")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> FeedForward:
        return cls(
            dim=int(config["hidden_size"]),
            hidden_dim=(
                int(config["ffn_hidden_size"])
                if config.get("ffn_hidden_size") is not None
                else None
            ),
            expansion_factor=float(config.get("ffn_expansion_factor", 4.0)),
            multiple_of=int(config.get("ffn_multiple_of", 1)),
            activation=str(config.get("ffn_activation", "gelu")),
            dropout=float(config.get("ffn_dropout", 0.0)),
            bias=bool(config.get("ffn_bias", True)),
            initializer_range=float(config.get("initializer_range", 0.02)),
            device=device,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden_dim={self.hidden_dim}, "
            f"activation={self.activation_name!r}, dropout={self.dropout_probability}, "
            f"gated={self.is_gated}"
        )

    def _validate_hidden_states(self, hidden_states: Tensor) -> None:
        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim < 2:
            raise ValueError("hidden_states must have at least two dimensions")
        if hidden_states.shape[-1] != self.dim:
            raise ValueError(
                f"hidden_states final dimension must be {self.dim}, "
                f"got {hidden_states.shape[-1]}"
            )
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must use a floating-point dtype")

    @staticmethod
    def _validate_configuration(
        dim: int,
        hidden_dim: int | None,
        expansion_factor: float,
        multiple_of: int,
        activation: str,
        dropout: float,
        initializer_range: float,
    ) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be an integer")
        if dim < 1:
            raise ValueError("dim must be positive")
        if hidden_dim is not None:
            if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool):
                raise TypeError("hidden_dim must be an integer or None")
            if hidden_dim < 1:
                raise ValueError("hidden_dim must be positive")
        if not math.isfinite(expansion_factor) or expansion_factor <= 0:
            raise ValueError("expansion_factor must be finite and positive")
        if not isinstance(multiple_of, int) or isinstance(multiple_of, bool):
            raise TypeError("multiple_of must be an integer")
        if multiple_of < 1:
            raise ValueError("multiple_of must be positive")
        if activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(SUPPORTED_ACTIVATIONS)}, got {activation!r}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not math.isfinite(initializer_range) or initializer_range <= 0:
            raise ValueError("initializer_range must be finite and positive")


class SparseMoE(nn.Module):
    """Top-k sparse mixture of feed-forward experts with capacity controls.

    The default configuration preserves the historical unlimited-capacity
    behaviour. ``capacity_factor`` can be enabled for production training to
    bound per-expert work and expose overflow diagnostics without changing the
    checkpoint layout.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        *,
        num_experts: int,
        experts_per_token: int = 2,
        expansion_factor: float = 4.0,
        multiple_of: int = 1,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        router_bias: bool = False,
        router_jitter: float = 0.0,
        capacity_factor: float | None = None,
        min_capacity: int = 0,
        initializer_range: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts < 1:
            raise ValueError("num_experts must be a positive integer")
        if (
            not isinstance(experts_per_token, int)
            or isinstance(experts_per_token, bool)
            or not 1 <= experts_per_token <= num_experts
        ):
            raise ValueError("experts_per_token must be between 1 and num_experts")
        if not math.isfinite(router_jitter) or router_jitter < 0:
            raise ValueError("router_jitter must be finite and non-negative")
        if capacity_factor is not None and (
            not math.isfinite(capacity_factor) or capacity_factor <= 0
        ):
            raise ValueError("capacity_factor must be finite and positive, or None")
        if not isinstance(min_capacity, int) or isinstance(min_capacity, bool) or min_capacity < 0:
            raise ValueError("min_capacity must be a non-negative integer")

        self.dim = dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.router_jitter = float(router_jitter)
        self.capacity_factor = float(capacity_factor) if capacity_factor is not None else None
        self.min_capacity = min_capacity
        self.router = nn.Linear(dim, num_experts, bias=router_bias, device=device, dtype=dtype)
        nn.init.normal_(self.router.weight, mean=0.0, std=initializer_range)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
        self.experts = nn.ModuleList(
            FeedForward(
                dim,
                hidden_dim=hidden_dim,
                expansion_factor=expansion_factor,
                multiple_of=multiple_of,
                activation=activation,
                dropout=dropout,
                bias=bias,
                initializer_range=initializer_range,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_experts)
        )
        self.hidden_dim = self.experts[0].hidden_dim
        self.last_router_aux_loss: Tensor | None = None
        self.last_router_z_loss: Tensor | None = None
        self.last_router_entropy: float | None = None
        self.last_expert_load: tuple[float, ...] = ()
        self.last_expert_capacity: int | None = None
        self.last_dropped_route_fraction: float = 0.0

    def _capacity(self, token_count: int) -> int | None:
        if self.capacity_factor is None:
            return None
        expected = token_count * self.experts_per_token / self.num_experts
        return max(self.min_capacity, int(math.ceil(self.capacity_factor * expected)))

    def forward(self, hidden_states: Tensor) -> Tensor:
        self.experts[0]._validate_hidden_states(hidden_states)
        original_shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, self.dim)
        token_count = tokens.shape[0]
        if token_count == 0:
            zero = self.router.weight.sum() * 0.0
            self.last_router_aux_loss = zero
            self.last_router_z_loss = zero
            self.last_router_entropy = 0.0
            self.last_expert_load = tuple(0.0 for _ in range(self.num_experts))
            self.last_expert_capacity = self._capacity(0)
            self.last_dropped_route_fraction = 0.0
            return torch.zeros_like(hidden_states)

        router_inputs = tokens
        if self.training and self.router_jitter:
            noise = torch.empty_like(tokens).uniform_(
                1.0 - self.router_jitter, 1.0 + self.router_jitter
            )
            router_inputs = tokens * noise
        router_logits = self.router(router_inputs)
        router_probabilities = F.softmax(router_logits.float(), dim=-1)
        top_logits, top_experts = torch.topk(
            router_logits, self.experts_per_token, dim=-1
        )
        top_weights = F.softmax(top_logits.float(), dim=-1).to(tokens.dtype)

        # Switch-style load balancing plus the router z-loss used by modern MoE
        # training recipes. The module exposes both unweighted signals so the
        # training policy can choose coefficients without checkpoint changes.
        importance = router_probabilities.mean(dim=0)
        hard_routes = F.one_hot(top_experts, num_classes=self.num_experts).float()
        load = hard_routes.mean(dim=(0, 1))
        self.last_router_aux_loss = self.num_experts * torch.sum(importance * load)
        self.last_router_z_loss = torch.logsumexp(router_logits.float(), dim=-1).square().mean()

        capacity = self._capacity(token_count)
        self.last_expert_capacity = capacity
        output = torch.zeros_like(tokens)
        dropped_routes = 0
        total_routes = token_count * self.experts_per_token

        # Dispatch only selected tokens. With a capacity limit, retain the
        # strongest routes for each expert deterministically and renormalize the
        # kept weights per token so overflow does not silently shrink activations.
        kept_by_expert: list[tuple[Tensor, Tensor, Tensor]] = []
        kept_weight_sum = torch.zeros(token_count, device=tokens.device, dtype=tokens.dtype)
        for expert_index in range(self.num_experts):
            token_index, route_index = torch.where(top_experts == expert_index)
            if token_index.numel() == 0:
                kept_by_expert.append((token_index, route_index, top_weights.new_empty((0,))))
                continue
            weights = top_weights[token_index, route_index]
            if capacity is not None and token_index.numel() > capacity:
                keep_order = torch.argsort(weights, descending=True, stable=True)[:capacity]
                dropped_routes += int(token_index.numel() - capacity)
                token_index = token_index.index_select(0, keep_order)
                route_index = route_index.index_select(0, keep_order)
                weights = weights.index_select(0, keep_order)
            kept_weight_sum.index_add_(0, token_index, weights)
            kept_by_expert.append((token_index, route_index, weights))

        for expert_index, expert in enumerate(self.experts):
            token_index, _, weights = kept_by_expert[expert_index]
            if token_index.numel() == 0:
                continue
            denom = kept_weight_sum.index_select(0, token_index).clamp_min(
                torch.finfo(weights.dtype).eps
            )
            normalized_weights = (weights / denom).unsqueeze(-1)
            expert_output = expert(tokens.index_select(0, token_index))
            output.index_add_(0, token_index, expert_output * normalized_weights)

        with torch.no_grad():
            entropy = -(
                router_probabilities
                * router_probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1).mean()
            self.last_router_entropy = float(entropy)
            self.last_expert_load = tuple(float(value) for value in load)
            self.last_dropped_route_fraction = dropped_routes / max(total_routes, 1)
        return output.reshape(original_shape)

    def routing_metrics(self) -> dict[str, object]:
        """Return detached routing diagnostics suitable for logs/telemetry."""
        return {
            "router_entropy": self.last_router_entropy,
            "expert_load": self.last_expert_load,
            "expert_capacity": self.last_expert_capacity,
            "dropped_route_fraction": self.last_dropped_route_fraction,
            "router_z_loss": (
                float(self.last_router_z_loss.detach())
                if self.last_router_z_loss is not None
                else None
            ),
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden_dim={self.hidden_dim}, num_experts={self.num_experts}, "
            f"experts_per_token={self.experts_per_token}, capacity_factor={self.capacity_factor}"
        )
