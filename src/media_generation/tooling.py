"""JSON schemas for exposing media generation as bounded agent tools."""
from __future__ import annotations

MEDIA_TOOL_SCHEMAS = {
    "generate_audio": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1, "maxLength": 4096},
            "negative_prompt": {"type": "string", "maxLength": 4096},
            "seconds": {"type": "number", "minimum": 0.1, "maximum": 300},
            "preset": {"type": "string", "enum": ["draft", "balanced", "quality", "max_quality"]},
            "style": {"type": "string"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
    "generate_video": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1, "maxLength": 4096},
            "negative_prompt": {"type": "string", "maxLength": 4096},
            "preset": {"type": "string", "enum": ["draft", "balanced", "quality", "max_quality"]},
            "style": {"type": "string"},
            "camera": {"type": "string", "enum": ["static", "dolly_in", "dolly_out", "pan_left", "pan_right", "orbit", "handheld", "drone"]},
            "segments": {"type": "integer", "minimum": 1, "maximum": 16},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}
