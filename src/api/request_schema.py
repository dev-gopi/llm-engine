"""API request contracts shared by Chat Completions and Responses."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from schema.structured_outputs import make_spec, StructuredSchemaError

class JSONSchemaResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["json_schema"] = "json_schema"
    json_schema: dict[str, Any]

    @field_validator("json_schema")
    @classmethod
    def validate_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        try: make_spec(name=str(value.get("name", "schema")), schema=value.get("schema", value), strict=bool(value.get("strict", False)))
        except StructuredSchemaError as exc: raise ValueError(str(exc)) from exc
        return value

class APIResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "json_object", "json_schema"]
    json_schema: dict[str, Any] | None = None
