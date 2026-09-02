# Model usage and serving guide

> The filename is retained for existing links. Commands use the current
> unversioned configs and the active `data/tokenizer-v2` artifact.

## Choose a checkpoint

Use the most advanced stage that has passed validation and behavioral checks:

1. `checkpoints/dpo/best.pt` — preference-aligned model.
2. `checkpoints/finetuning/best.pt` — supervised assistant model.
3. `checkpoints/pretraining/best.pt` — raw completion model.

Do not select a checkpoint only because it is newest. `best.pt` is chosen by
validation; `latest.pt` exists primarily for resuming interrupted training.

## Generate from the CLI

```bash
.venv/bin/python scripts/generate.py "Explain gradient accumulation simply." \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/dpo/best.pt \
  --device cuda
```

Use `configs/model.cpu.yaml` and `--device cpu` only with a checkpoint and
tokenizer compatible with that CPU model shape.

## Terminal chat

```bash
.venv/bin/python scripts/chat.py \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/dpo/best.pt \
  --device cuda
```

The inference config controls sampling, context memory, web search, RAG, rate
limits, and serving defaults.

## Browser UI and API

Set the active paths explicitly when they differ from `configs/inference.yaml`:

```bash
export GOPI_INFERENCE_CONFIG=configs/inference.yaml
export GOPI_MODEL_CONFIG=configs/model.gpu.yaml
export GOPI_TOKENIZER_PATH=data/tokenizer-v2
export GOPI_CHECKPOINT_PATH=checkpoints/dpo/best.pt
export GOPI_MODEL_NAME=gopi
export GOPI_API_KEY='replace-with-a-secret'
```

Launch the API using the project serving entry point documented in
[DEPLOYMENT.md](DEPLOYMENT.md), then use the browser UI or an OpenAI-compatible
client with base URL `http://HOST:8000/v1`, model `gopi`, and the configured API
key.

## Non-streaming request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GOPI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gopi",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

## Streaming request

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GOPI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gopi",
    "messages": [{"role": "user", "content": "Write a short greeting"}],
    "stream": true
  }'
```

Treat retrieved web and RAG content as untrusted context. Keep authentication,
TLS, request limits, and storage permissions appropriate for the deployment.

## Export and run portable weights

```bash
.venv/bin/python scripts/export.py \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/dpo/best.pt \
  --format safetensors \
  --output exports/final/gopi.safetensors
```

```bash
.venv/bin/python scripts/generate_exported.py "Hello" \
  --model exports/final/gopi.safetensors \
  --model-config exports/final/model.yaml \
  --tokenizer exports/final/tokenizer \
  --device cuda
```

An export is a set: weights, `model.yaml`, and tokenizer must remain together.

## Troubleshooting

- Vocabulary mismatch: use the tokenizer that trained the checkpoint or a
  verified append-only descendant.
- Shape mismatch: use the exact model config stored with the checkpoint.
- CUDA/BF16 error: select compatible hardware or an explicit CPU profile.
- Poor assistant behavior from a pretraining checkpoint: use the best SFT or
  DPO checkpoint instead.
- Old live-report points after a new stage: update to the current `train.py`;
  new stages archive old report files, while `--resume` preserves history.
