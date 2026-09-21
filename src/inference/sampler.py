"""Top-k/top-p token sampling."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .temperature import apply_temperature


class TopKSampler:
    def __call__(
        self,
        logits: Tensor,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if not isinstance(logits, Tensor) or not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor")
        if logits.ndim != 2 or logits.size(-1) == 0:
            raise ValueError("logits must have shape [batch, vocabulary]")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        if not math.isfinite(min_p) or not 0 <= min_p <= 1:
            raise ValueError("min_p must satisfy 0 <= min_p <= 1")
        # Validate both greedy and stochastic paths. Negative infinity is a
        # legitimate blocked token, but each row must retain a usable token.
        if torch.isnan(logits).any() or torch.isposinf(logits).any():
            raise ValueError("logits cannot contain NaN or positive infinity")
        if not torch.isfinite(logits).any(dim=-1).all():
            raise ValueError("each logits row must contain an unblocked finite token")
        filtered = apply_temperature(logits.float(), temperature)
        if temperature == 0:
            return logits.argmax(dim=-1)
        if min_p:
            # p(token) >= min_p * p(best), expressed in logit space.
            threshold = filtered.amax(dim=-1, keepdim=True) + math.log(min_p)
            filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
        if top_k:
            k = min(top_k, filtered.size(-1))
            threshold = filtered.topk(k, dim=-1).values[:, -1, None]
            filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = filtered.sort(dim=-1, descending=True)
            cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(filtered, float("-inf")).scatter(
                1, sorted_indices, sorted_logits
            )
        return torch.multinomial(filtered.softmax(dim=-1), 1, generator=generator).squeeze(1)

class TokenConstraint:
    """Interface for token-time constraints used before sampling."""

    def filter_logits(self, logits: Tensor, generated_ids: list[int], tokenizer, *, candidate_k: int = 256) -> Tensor:
        raise NotImplementedError

    def validate(self, text: str) -> bool:
        return True


class PrefixGrammarConstraint(TokenConstraint):
    """Generic deterministic grammar callback.

    The callback receives decoded text and returns the token strings that may be
    appended. This is intentionally tokenizer-aware and keeps the sampler
    independent of a particular grammar implementation.
    """

    def __init__(self, allowed_next: callable, *, final_validator: callable | None = None) -> None:
        self.allowed_next = allowed_next
        self.final_validator = final_validator

    def filter_logits(self, logits: Tensor, generated_ids: list[int], tokenizer, *, candidate_k: int = 256) -> Tensor:
        if logits.ndim != 2 or logits.size(0) != 1:
            raise ValueError("token constraints currently require logits with batch size 1")
        prefix = tokenizer.decode(generated_ids, skip_special_tokens=False)
        allowed = set(self.allowed_next(prefix))
        if not allowed:
            raise ValueError("grammar produced no allowed token strings")
        values, ids = torch.topk(logits, min(max(1, candidate_k), logits.size(-1)), dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        for token_id in ids[0].tolist():
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            if piece in allowed:
                mask[0, int(token_id)] = logits[0, int(token_id)]
        if not torch.isfinite(mask).any():
            # Fall back to an exhaustive vocabulary pass only when the shortlist
            # contains no legal token; this preserves correctness without making
            # the common path O(vocabulary) per decoding step.
            for token_id in range(logits.size(-1)):
                piece = tokenizer.decode([token_id], skip_special_tokens=False)
                if piece in allowed:
                    mask[0, token_id] = logits[0, token_id]
        if not torch.isfinite(mask).any():
            raise ValueError("grammar rejected every token")
        return mask

    def validate(self, text: str) -> bool:
        return bool(self.final_validator(text)) if self.final_validator is not None else True


class JSONSchemaConstraint(TokenConstraint):
    """Incrementally constrain generation to structurally valid JSON.

    The supported schema validator intentionally covers the common object schema
    subset (type/object/array/string/number/integer/boolean/null, required, and
    properties). Generation is prefix-constrained; final schema validation remains
    mandatory as defense in depth.
    """

    def __init__(self, schema: dict) -> None:
        if not isinstance(schema, dict):
            raise TypeError("json schema must be a mapping")
        self.schema = schema

    @staticmethod
    def _prefix_valid(text: str) -> bool:
        in_string = False
        escaped = False
        stack: list[str] = []
        for char in text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char == "}" and (not stack or stack.pop() != "{"):
                return False
            elif char == "]" and (not stack or stack.pop() != "["):
                return False
        if escaped or in_string:
            return not escaped
        # A closing structural token cannot immediately follow a colon or comma.
        stripped = text.rstrip()
        if stripped.endswith((":", ",")) or stripped.endswith((":}", ",}")) or stripped.endswith((":]", ",]")):
            return False
        return not escaped

    def _schema_type_valid(self, value, schema: dict) -> bool:
        expected = schema.get("type")
        if expected is None:
            return True
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        if expected == "string":
            return isinstance(value, str)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "null":
            return value is None
        return True

    def validate(self, text: str) -> bool:
        import json
        try:
            value = json.loads(text)
        except (TypeError, ValueError):
            return False
        if not self._schema_type_valid(value, self.schema):
            return False
        if isinstance(value, dict) and self.schema.get("type") == "object":
            required = self.schema.get("required", [])
            if any(key not in value for key in required):
                return False
            for key, property_schema in self.schema.get("properties", {}).items():
                if key in value and isinstance(property_schema, dict) and not self._schema_type_valid(value[key], property_schema):
                    return False
        return True

    def filter_logits(self, logits: Tensor, generated_ids: list[int], tokenizer, *, candidate_k: int = 256) -> Tensor:
        if logits.ndim != 2 or logits.size(0) != 1:
            raise ValueError("JSON constraints currently require logits with batch size 1")
        prefix = tokenizer.decode(generated_ids, skip_special_tokens=False)
        eos_id = tokenizer.token_to_id("<|eos|>")
        values, ids = torch.topk(logits, min(max(1, candidate_k), logits.size(-1), dim=-1))
        mask = torch.full_like(logits, float("-inf"))
        import json
        for token_id in ids[0].tolist():
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            candidate = prefix + piece
            if eos_id is not None and int(token_id) == eos_id:
                if self.validate(prefix):
                    mask[0, int(token_id)] = logits[0, int(token_id)]
                continue
            if not self._prefix_valid(candidate):
                continue
            try:
                parsed = json.loads(candidate)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and not self.validate(candidate):
                continue
            mask[0, int(token_id)] = logits[0, int(token_id)]
        if not torch.isfinite(mask).any():
            for token_id in range(logits.size(-1)):
                piece = tokenizer.decode([token_id], skip_special_tokens=False)
                candidate = prefix + piece
                if self._prefix_valid(candidate):
                    mask[0, token_id] = logits[0, token_id]
        if not torch.isfinite(mask).any():
            raise ValueError("JSON schema rejected every token")
        return mask
