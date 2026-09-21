"""Strict JSON-Schema structured-output validation and compatibility contracts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import Draft202012Validator, SchemaError, ValidationError


class StructuredOutputError(ValueError):
    code = "invalid_structured_output"


class StructuredSchemaError(StructuredOutputError):
    code = "invalid_json_schema"


class StructuredOutputValidationError(StructuredOutputError):
    code = "structured_output_validation_failed"


@dataclass(frozen=True)
class StructuredOutputSpec:
    name: str
    schema: dict[str, Any]
    strict: bool = False


def validate_schema_compatibility(schema: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        raise StructuredSchemaError("json_schema must be an object")
    normalized = dict(schema)
    try:
        validator = Draft202012Validator.check_schema(normalized)
    except SchemaError as exc:
        raise StructuredSchemaError(f"malformed JSON Schema: {exc.message}") from exc
    # The runtime guarantees deterministic JSON values, not arbitrary external
    # formats or executable schema extensions.
    if normalized.get("$dynamicRef") or normalized.get("$recursiveRef"):
        raise StructuredSchemaError("dynamic/recursive schema references are not supported")
    return normalized


def validate_structured_output(text: str, spec: StructuredOutputSpec) -> Any:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructuredOutputValidationError("generated output is not valid JSON") from exc
    validator = Draft202012Validator(spec.schema)
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        location = ".".join(str(p) for p in first.path) or "$"
        raise StructuredOutputValidationError(f"schema violation at {location}: {first.message}")
    return value


def make_spec(*, name: str, schema: Mapping[str, Any], strict: bool = False) -> StructuredOutputSpec:
    if not isinstance(name, str) or not name.strip():
        raise StructuredSchemaError("json_schema.name must be non-empty")
    if len(name) > 64:
        raise StructuredSchemaError("json_schema.name cannot exceed 64 characters")
    normalized = validate_schema_compatibility(schema)
    if strict:
        # Strict mode requires closed objects unless the schema explicitly opts
        # into a non-object root. This prevents accidental extra fields.
        def walk(node: Any) -> None:
            if isinstance(node, Mapping):
                if node.get("type") == "object" and node.get("additionalProperties", False) is not False:
                    raise StructuredSchemaError("strict object schemas must set additionalProperties=false")
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(normalized)
    return StructuredOutputSpec(name=name.strip(), schema=normalized, strict=bool(strict))
