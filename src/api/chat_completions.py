"""Structured Chat Completions response helpers."""
from __future__ import annotations

from typing import Any

from schema.structured_outputs import (
    StructuredOutputSpec,
    make_spec,
    validate_structured_output,
)


def parse_response_format(response_format: Any) -> StructuredOutputSpec | None:
    if response_format is None or getattr(response_format, "type", None) != "json_schema": return None
    payload = getattr(response_format, "json_schema", None)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(by_alias=True, mode="json")
    if not isinstance(payload, dict): raise ValueError("response_format.json_schema is required")
    schema = payload.get("schema")
    if not isinstance(schema, dict): raise ValueError("response_format.json_schema.schema is required")
    return make_spec(name=str(payload.get("name", "response")), schema=schema, strict=bool(payload.get("strict", False)))

def validate_response_text(text: str, spec: StructuredOutputSpec | None) -> Any:
    return validate_structured_output(text, spec) if spec else None
