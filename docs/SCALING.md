# Configuration-driven model scaling

MiniGPT already reads layer count, hidden size, attention/KV heads, feed-forward
size, vocabulary, context length, normalization, and positional encoding from
YAML. A new architecture can inherit a profile through `extends` and override
those values without editing the model implementation.

`configs/scaling/model.1t.yaml` describes 999,772,348,416 dense parameters with
40,000 vocabulary entries: 164 layers, width 24,576, 192 attention heads,
8 KV heads, and a 65,536-wide SwiGLU intermediate layer. This is a planning
example, not an architecture selected through quality experiments.

Inspect it without allocating model weights:

```bash
.venv/bin/python scripts/inspect_model.py configs/scaling/model.1t.yaml
.venv/bin/python scripts/plan_training.py \
  --model-config configs/scaling/model.1t.yaml \
  --training-config configs/scaling/training.1t.yaml \
  --training-tokens 1000000000000 --gpus 1024
```

The token and GPU counts above are illustrative planning inputs, not training
recommendations. The planner is a rough analytical estimator. It does not model
initialization peaks, full-layer FSDP materialization, communication topology,
or vocabulary projection/loss memory in sufficient detail to prove feasibility.

## What is still required

The current trainer constructs the full model on each device before FSDP
wrapping. Approximately one trillion FP32 parameters alone require about 4 TB
in decimal units, before gradients or optimizer state. Changing GPU count in
a configuration cannot fix this initialization path.

A production runtime at this scale needs sharded/meta initialization and
checkpoint loading, an appropriate model-parallel training strategy, validated
memory estimates including gathered layers, and cluster-scale failure/recovery
and throughput testing. Training tensor/pipeline parallelism is not currently
wired into this training entry point. The existing inference tensor-parallel
feature does not provide that training capability.

Both templates are marked `planning_only: true`; the model factory and training
entry point reject them before weight allocation. Removing the flag does not
solve these runtime limitations. This repository does not yet promise runnable
one-trillion-parameter training or serving through configuration alone.

Changing dimensions also changes checkpoint shapes. The current small checkpoint
cannot directly initialize this profile with `--init-from`. A future training
run needs a compatible checkpoint or a separately validated weight-growth
procedure, a matching tokenizer, and sufficient training data. Increasing
parameter count alone does not establish better output quality.
