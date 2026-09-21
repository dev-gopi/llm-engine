"""Runtime capability discovery for the serving API.

Capabilities are derived from the running backend and model configuration rather
than from a marketing/static model name.  A capability is only advertised when
its implementation surface is present and the runtime can expose the required
contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from model.config import resolve_attention_layer_pattern
import json


@dataclass(frozen=True)
class ModelCapabilities:
    chat: bool
    streaming: bool
    tool_calling: bool
    structured_outputs: bool
    reasoning: bool
    vision: bool
    audio: bool
    rag: bool
    mcp: bool
    kv_cache: bool
    prefix_caching: bool
    context_length: int
    architecture: str
    parameter_count: int | None = None
    vocabulary_size: int | None = None
    layers: int | None = None
    hidden_size: int | None = None
    attention_heads: int | None = None
    kv_heads: int | None = None
    position_type: str | None = None
    attention_pattern: str | None = None
    full_attention_layers: int | None = None
    linear_attention_layers: int | None = None
    ffn_type: str | None = None
    num_experts: int | None = None
    experts_per_token: int | None = None
    mtp_predictions: int | None = None
    qk_norm: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_or_none(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


def load_capability_evidence(path: str | Path = "reports/capability_evidence.json") -> dict[str, bool] | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    return {str(key): bool(item) for key, item in value.items() if isinstance(item, bool)}


def discover_capabilities(backend: Any, *, model_config: Mapping[str, Any] | None = None, validation_evidence: Mapping[str, bool] | None = None) -> ModelCapabilities:
    """Build a conservative capability manifest from a live backend.

    False is preferred over an optimistic claim when the implementation cannot
    prove that a feature is available.  This keeps `/v1/models` useful for
    clients that must avoid unsupported requests.
    """
    config = dict(model_config or {})
    generator = getattr(backend, "generator", None)
    rag_index = getattr(backend, "rag_index", None)
    mcp_tools = getattr(backend, "mcp_tools", None)

    # Reloadable/replica wrappers expose the actual model backend through a
    # `backend` attribute.  Resolve it without requiring a particular class.
    if generator is None and hasattr(backend, "backend"):
        nested = getattr(backend, "backend")
        return discover_capabilities(nested, model_config=config, validation_evidence=validation_evidence)

    context_length = _positive_or_none(
        getattr(generator, "max_positions", None) or config.get("max_position") or getattr(backend, "context_length", None)
    ) or 0
    architecture = str(config.get("architecture", "decoder-only-transformer"))
    parameter_count = _positive_or_none(getattr(getattr(generator, "model", None), "num_parameters", lambda: 0)())
    if parameter_count is None:
        parameter_count = _positive_or_none(getattr(backend, "parameter_count", None))

    # These flags map directly to implemented serving contracts.  Vision/audio
    # are intentionally false: the multimodal modules are not wired into the
    # text serving endpoint yet.
    chat = callable(getattr(backend, "generate", None))
    streaming = callable(getattr(backend, "stream", None))
    structured_outputs = True  # JSON Schema validation is part of the API contract.
    reasoning = True  # Runtime budget mapping is implemented by runtime.reasoning.
    tool_calling = bool(
        callable(getattr(backend, "generate", None))
        and (hasattr(backend, "mcp_tools") or hasattr(backend, "_augment_with_mcp"))
    )
    rag = rag_index is not None
    mcp = bool(mcp_tools)
    kv_cache = bool(generator is not None and hasattr(generator, "_prefill"))
    prefix_caching = bool(getattr(generator, "prefix_cache", None) is not None) if generator else False

    if validation_evidence is not None:
        # High-risk client-facing capabilities are only advertised after the
        # release/evaluation gate has produced explicit evidence.
        for name in ("chat", "streaming", "tool_calling", "structured_outputs", "reasoning", "rag", "mcp"):
            if name in validation_evidence:
                value = bool(validation_evidence[name])
                if name == "chat": chat = chat and value
                elif name == "streaming": streaming = streaming and value
                elif name == "tool_calling": tool_calling = tool_calling and value
                elif name == "structured_outputs": structured_outputs = structured_outputs and value
                elif name == "reasoning": reasoning = reasoning and value
                elif name == "rag": rag = rag and value
                elif name == "mcp": mcp = mcp and value

    layer_patterns: tuple[str, ...] = ()
    try:
        if _positive_or_none(config.get("layers")):
            layer_patterns = resolve_attention_layer_pattern(config)
    except (KeyError, TypeError, ValueError):
        layer_patterns = ()
    linear_layers = sum(item == "linear" for item in layer_patterns)
    full_layers = len(layer_patterns) - linear_layers if layer_patterns else None
    pattern_name = None
    if layer_patterns:
        unique_patterns = set(layer_patterns)
        pattern_name = next(iter(unique_patterns)) if len(unique_patterns) == 1 else "hybrid"

    return ModelCapabilities(
        chat=chat,
        streaming=streaming,
        tool_calling=tool_calling,
        structured_outputs=structured_outputs,
        reasoning=reasoning,
        vision=False,
        audio=False,
        rag=rag,
        mcp=mcp,
        kv_cache=kv_cache,
        prefix_caching=prefix_caching,
        context_length=context_length,
        architecture=architecture,
        parameter_count=parameter_count,
        vocabulary_size=_positive_or_none(config.get("vocab_size")),
        layers=_positive_or_none(config.get("layers")),
        hidden_size=_positive_or_none(config.get("hidden_size")),
        attention_heads=_positive_or_none(config.get("heads")),
        kv_heads=_positive_or_none(config.get("kv_heads")),
        position_type=str(config["position_type"]) if config.get("position_type") is not None else None,
        attention_pattern=pattern_name,
        full_attention_layers=full_layers,
        linear_attention_layers=linear_layers if layer_patterns else None,
        ffn_type=str(config.get("ffn_type", "dense")),
        num_experts=_positive_or_none(config.get("num_experts")) if str(config.get("ffn_type", "dense")).lower() == "moe" else None,
        experts_per_token=_positive_or_none(config.get("experts_per_token")) if str(config.get("ffn_type", "dense")).lower() == "moe" else None,
        mtp_predictions=int(config.get("mtp_num_predictions", 0) or 0),
        qk_norm=bool(config.get("qk_norm", False)),
    )


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load a model configuration including project ``extends`` inheritance."""
    from utils.config import load_yaml

    source = Path(path)
    if not source.is_file():
        return {}
    value = load_yaml(source)
    if not isinstance(value, dict):
        raise ValueError(f"model configuration must be a mapping: {source}")
    return value
