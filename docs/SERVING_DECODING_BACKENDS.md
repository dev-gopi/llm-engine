# Serving decoding backends

The native serving request supports three decoding strategies:

- `sample` (default): normal temperature/top-k/top-p generation and token streaming.
- `beam`: deterministic beam search using `num_beams` and `length_penalty`. Streaming responses are buffered until the beam search completes.
- `speculative`: native speculative decoding against a configured draft checkpoint. Streaming responses are buffered until verification completes.

Configure one native draft model under `serving` in `configs/inference.yaml` or with:

```bash
export GOPI_SPECULATIVE_DRAFT_MODEL_ID=draft-v1
export GOPI_SPECULATIVE_DRAFT_CHECKPOINT=checkpoints/draft/best.pt
export GOPI_SPECULATIVE_DRAFT_MODEL_CONFIG=configs/model.draft.yaml
```

A speculative request must provide the matching `draft_model_id`. Unknown draft ids fail explicitly.

## vLLM backend

The server can use the optional in-process vLLM backend:

```bash
export GOPI_BACKEND=vllm
export GOPI_VLLM_MODEL=/path/to/model-or-hf-id
export GOPI_VLLM_TENSOR_PARALLEL_SIZE=1
# optional: GOPI_VLLM_DTYPE=bfloat16
# optional: GOPI_VLLM_MAX_MODEL_LEN=4096
```

The vLLM dependency remains optional. If it is not installed, startup fails explicitly rather than silently falling back to the native backend. The current stable adapter uses vLLM's synchronous `LLM` API and therefore exposes buffered streaming.

## GGUF export

The main export CLI now exposes the built-in GGUF v3 writer:

```bash
python scripts/export.py --format gguf --weight-dtype float16 --output exports/gopi.gguf
```

This produces an F16/F32 GGUF container. It is not a claim that every MiniGPT architecture is directly loadable by llama.cpp, and it does not replace target-runtime quantization/architecture metadata work.
