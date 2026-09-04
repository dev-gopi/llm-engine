# Capabilities, limitations, and model scaling

> The filename is retained for existing links. This document describes the
> current unversioned profiles.

## Language-model stability features

New from-scratch model profiles can enable parameter-free stability options:

```yaml
qk_norm: true
qk_norm_eps: 0.000001
logit_softcap: 30.0
```

`qk_norm` normalizes each projected query and key head before rotary embeddings
and attention. `logit_softcap` smoothly bounds vocabulary logits with `tanh` for
safer mixed-precision numerics and less extreme confidence. These options change
model behavior, so use them for newly trained checkpoints rather than silently
adding them to an existing checkpoint configuration.

## Active model

`configs/model.gpu.yaml` defines the active laptop-GPU architecture:

- 40,000-token base vocabulary, expandable to the verified 42,000-token tokenizer;
- hidden size 512;
- 16 transformer layers;
- 8 attention heads and 2 KV heads;
- rotary positions with a 512-token model limit;
- RMSNorm, SwiGLU, tied embeddings, and gradient checkpointing.

It contains about 80.3M parameters. `configs/model.cpu.yaml` remains a smaller
32K/8-layer profile and is not checkpoint-compatible with the active GPU model.

Inspect any architecture without allocating its tensors:

```bash
python scripts/inspect_model.py configs/model.gpu.yaml
python scripts/inspect_model.py configs/model.cpu.yaml
```

## Realistic capabilities

After successful pretraining, SFT, and evaluation, the active model can support
small-project experiments in:

- short chat and instruction following;
- English, Bengali, Hindi, and Hinglish text represented in the training data;
- basic coding and editing tasks;
- GSM8K-style arithmetic patterns;
- safety-response, tool-call, and structured-output experiments;
- local RAG and web-search demonstrations;
- checkpoint growth, recovery SFT, DPO, export, and serving research.

These are expected task categories, not guaranteed quality claims. Actual
capability depends on dataset quality, training tokens, convergence, and held-out
evaluation.

## Important limitations

An 80M model trained on the included educational corpus is not comparable to a
large production foundation model. Expect limitations in factual reliability,
multi-step reasoning, long-context consistency, code correctness, multilingual
fluency, tool selection, and resistance to prompt injection. RAG and web search
provide context; they do not guarantee truth.

Never treat model output as authoritative medical, legal, financial, security,
or safety-critical advice. Validate generated code and factual claims.

## What each stage contributes

| Stage | Main purpose | Does not guarantee |
| --- | --- | --- |
| `checkpoints/pretraining/best.pt` | Language modeling and broad corpus patterns | Assistant behavior |
| `checkpoints/finetuning/best.pt` | Instructions, chat, multilingual, math, coding, and safety behavior | Preference alignment or factuality |
| `checkpoints/recovery/best.pt` | Focused repair of response quality | Broad new knowledge |
| `checkpoints/dpo/best.pt` | Preference toward chosen responses | Correct answers outside learned data |

Use `best.pt` for comparison and deployment. Use `latest.pt` to resume an
interrupted stage.

## Scaling profiles

Future architecture targets live under `configs/text/`:

- `model.future.1b.yaml` — roughly 1.185B parameters and 8K context;
- `model.future.7b.yaml` — multi-GPU/multi-node target with 32K context;
- `model.future.30b.yaml` — cluster-scale target with 32K context;
- `tokenizer.future.50k.yaml` — separate 50K from-scratch tokenizer family;
- `pretraining.future.fsdp.yaml` — opt-in FSDP training example.

The future tokenizer is intentionally incompatible with the active 40K model.
Using it requires new pretraining.

```bash
python scripts/inspect_model.py configs/text/model.future.1b.yaml
python scripts/inspect_model.py configs/text/model.future.7b.yaml
python scripts/inspect_model.py configs/text/model.future.30b.yaml
```

## Data and compute requirements

Increasing parameter count without increasing unique, reviewed data usually
increases memorization rather than capability. Before scaling:

1. Deduplicate train/validation/test data and check contamination.
2. Record source, version, license, allowed use, and privacy review.
3. Measure unique tokens and domain balance.
4. Validate mixed precision, checkpoint restore, and distributed communication.
5. Estimate optimizer, gradient, activation, KV-cache, and temporary memory—not
   only parameter weights.
6. Reserve stable evaluations and fixed generation prompts.

Plan a run before allocating hardware:

```bash
.venv/bin/python scripts/plan_training.py \
  --model-config configs/text/model.future.1b.yaml \
  --training-config configs/text/pretraining.future.fsdp.yaml \
  --training-tokens 20000000000 \
  --hardware-tflops 100 \
  --utilization 0.35 \
  --gpu-memory-gib 24 \
  --require-fit
```

The planner is an estimate. Validate throughput and memory on the real cluster
before committing to a long run.

## Recommended progression

1. Make the active 80M route reproducible.
2. Compare pretraining, SFT, recovery, and DPO using fixed evaluations.
3. Test intermediate 100M–350M configurations.
4. Validate FSDP and distributed checkpoint resharding on a short run.
5. Move to the 1B profile only when data and compute scale with it.
6. Treat 7B and 30B as cluster projects requiring production-grade operations.

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training commands and
[USAGE_GUIDE.md](USAGE_GUIDE.md) for inference and serving.
