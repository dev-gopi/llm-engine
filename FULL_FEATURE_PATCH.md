# Gopi Chat-Completions Feature Patch

This patch completes the serving/runtime plumbing for OpenAI-compatible chat completions with tool calling, multimodal image input, explicit reasoning output, and streaming.

## Implemented

- `POST /v1/chat/completions` accepts OpenAI-style text or `image_url` content parts.
- Standard OpenAI function definitions, `tool_choice` (`none`, `auto`, `required`, or a specific function), schema validation, native tool-call parsing, `finish_reason: "tool_calls"`, and streamed/non-streamed `tool_calls`.
- `reasoning_effort` and separation of a native `<thinking>...</thinking>` trace into `reasoning_content` while returning only the final answer as `content`.
- SSE streaming for normal text plus protocol-safe reasoning and tool-call events.
- Optional native vision serving through the existing `VisionEncoder -> VisionProjector -> MiniGPT` path when trained multimodal weights are configured.
- Delegated OpenAI-compatible backends now forward multimodal messages, tools, tool choice, and reasoning and parse them on the way back.
- Runtime capability reporting now includes `tool_calling`, `reasoning`, `vision`, `max_input_tokens`, and `max_output_tokens` based on the loaded backend/model.
- Remote image fetching is disabled by default and, when explicitly enabled, rejects private/local addresses and redirects.

## Native vision activation

The uploaded archive did not contain trained model/tokenizer or multimodal checkpoint files. Native vision therefore remains capability-gated until trained artifacts are supplied.

Configure either `configs/inference.yaml` or environment variables:

```yaml
serving:
  multimodal_config: configs/vision/multimodal.yaml
  multimodal_checkpoint: checkpoints/vision/multimodal-best.pt
  vision_checkpoint: null
  vision_allow_remote_images: false
  vision_max_image_bytes: 10485760
  vision_max_images: 4
```

Equivalent environment variables:

```bash
export GOPI_MULTIMODAL_CONFIG=configs/vision/multimodal.yaml
export GOPI_MULTIMODAL_CHECKPOINT=checkpoints/vision/multimodal-best.pt
# Optional only when vision encoder weights are separate:
# export GOPI_VISION_CHECKPOINT=checkpoints/vision/vision-best.pt
```

The multimodal checkpoint must contain trained `projector.*` weights and either `vision_encoder.*` weights or a separate usable vision checkpoint. The existing validated language-model checkpoint remains the text backbone.

## Context limits

This patch intentionally does **not** fake a 1,000,000-token context window or a 384,000-token output window. The serving API no longer has the old 8K request-schema ceiling, but the loaded model's real `max_position` is authoritative and is enforced at runtime.

The active GPU model config in this archive is still 1,024 tokens. The repository has long-context configurations up to 8,192 tokens. A genuine 1M context requires model/attention design, positional scaling, training, memory engineering, and long-context evaluation; changing a JSON number alone is not correct.

## Tests performed

- `python -m compileall -q src` — passed.
- Serving/API/conformance/reasoning/vision regression suite — **111 passed**.
- A complete `pytest -q` attempt was blocked during collection by missing `pyarrow` in the execution environment. `pyarrow>=15` is already declared by this project's `pyproject.toml`, but package download was unavailable because this environment has no package-network access.
- Running the remaining non-`pyarrow` project tests with the project `PYTHONPATH` reached 100% of test execution; the harness did not terminate before its execution timeout, so this is not reported as a clean full-suite exit.

## Accuracy note

The API/runtime implementation can be tested deterministically, but no code patch can guarantee 100% model accuracy for choosing tools, visual understanding, or reasoning quality. Those depend on the trained checkpoints and evaluation data. This archive contains no trained checkpoints, so end-to-end checkpoint accuracy could not be measured here.
