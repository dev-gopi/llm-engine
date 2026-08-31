# Troubleshooting guide

Run commands from the repository root with the virtual environment activated:

```bash
source .venv/bin/activate
python scripts/capabilities.py
pytest -q
```

## Non-finite gradients

A warning such as `gradient_norm=inf` or `gradient_norm=nan` means that one
optimizer update was discarded. With FP16, the gradient scaler then reduces
the loss scale. An isolated event is recoverable; repeated events indicate an
unstable run.

Check these values in subsequent log entries:

- `nonfinite_updates` should stop increasing;
- loss and gradient norm should return to finite values;
- validation loss should remain stable or improve.

On supported hardware, BF16 is usually more stable and does not use a gradient
scaler. Verify support before changing a configuration:

```bash
.venv/bin/python scripts/capabilities.py
```

If BF16 is unavailable, lower `learning_rate`, keep `gradient_clip_norm`
enabled, and use conservative `grad_scaler_initial_scale` and
`grad_scaler_growth_interval` settings. Do not restart from scratch for one
discarded update.

## CUDA out of memory

Reduce resource use in this order:

1. lower `batch_size`;
2. increase `gradient_accumulation_steps` to preserve the effective batch;
3. lower `max_sequence_length`;
4. enable gradient checkpointing in the model configuration;
5. use a packed-data profile to remove runtime tokenization overhead.

Before retrying, stop the failed process and confirm that no old training
process still owns GPU memory. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
can reduce allocator fragmentation, but it does not create additional VRAM.

## Tokenizer fingerprint mismatch

The selected tokenizer does not match the tokenizer recorded in the
checkpoint. Use the original tokenizer, or extend it only by appending tokens
while preserving every old token ID. A retrained tokenizer that changes old
IDs is incompatible with the checkpoint.

Use `--resume` only to continue the same run. Use `--init-from` when starting a
new stage such as pretraining to SFT or SFT to another experiment.

## Checkpoint does not load

Confirm that model architecture, tokenizer, and checkpoint belong together.
Do not use a CPU architecture configuration to load a GPU-profile checkpoint;
the names describe model shapes as well as intended hardware.

`latest.pt` stores the most recent resumable state. `best.pt` is selected by
validation and should normally initialize the next training stage or inference.

## Training loss changes after resume

Small batch-to-batch changes are normal. A resumed run restores optimizer and
scheduler state, so its learning rate should continue rather than restart.
Verify the logged epoch, step, learning rate, tokenizer fingerprint, and
validation metric name.

Changing data weights or the validation metric creates a different comparison
baseline. The trainer intentionally resets best-loss tracking when the metric
identity changes; this does not erase model weights.

## Validation gets worse

Compare validation checkpoints, not individual training batches. Inspect every
domain separately because an aggregate can hide regressions. Stop or rely on
`best.pt` when several evaluations fail to improve. If TinyStories improves but
WikiText worsens, rebalance the training mix instead of merely adding epochs.

## Generation repeats or becomes incoherent

First try a lower temperature, lower `top_p`, and a modest repetition penalty.
Ensure the prompt plus requested completion fits the model context window.
Persistent quality problems require better data, domain-balanced evaluation,
continued pretraining, or SFT; decoding settings cannot add missing knowledge.

## API reports not ready

Check:

```bash
curl -s http://127.0.0.1:8000/health/live
curl -s http://127.0.0.1:8000/health/ready
```

Verify `GOPI_CHECKPOINT_PATH`, `GOPI_MODEL_CONFIG`, `GOPI_TOKENIZER_PATH`, and
`GOPI_DEVICE`. Read the server traceback if readiness stays false.

## Third-party UI cannot connect

Use base URL `http://HOST:8000/v1`, the configured model name, and the bearer
key from `GOPI_API_KEY`. A container cannot reach the host through its own
`127.0.0.1`; use the Docker host-gateway address or an explicitly reachable
host address. Configure `GOPI_CORS_ORIGINS` only for browser origins you trust.

## Still unresolved

Capture the command, configuration paths, checkpoint step, last validation
block, full traceback, Python/PyTorch versions, GPU name, VRAM, and output from
`scripts/capabilities.py`. Remove secrets and private dataset text before
sharing logs.
