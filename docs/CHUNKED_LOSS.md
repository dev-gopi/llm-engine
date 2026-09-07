# Chunked language-model loss

`loss_chunk_size` is an optional training YAML setting. It limits the number of
valid target tokens processed together by the FP32 cross-entropy and z-loss
calculations. The main `configs/finetuning.gpu.yaml` profile enables it at 128.
Existing profiles that omit it keep the original behavior.

```yaml
loss_chunk_size: 128
```

During training, loss chunks use activation checkpointing: intermediate tensors
are recomputed during backward instead of retained for every target token.
Evaluation also processes chunks, without backward recomputation. Padding and
masked instruction tokens remain excluded. Label smoothing, z-loss, mean/sum
reduction, and shifted targets retain their existing semantics, subject to normal
floating-point rounding differences.

This reduces retained loss intermediates, not model weights, optimizer state,
EMA storage, or the full logits produced by the model. Extra recomputation and
per-chunk gradient work can slow training. Smaller chunks save intermediate
memory but add overhead. Set `loss_chunk_size: 0` to disable the feature; compare
peak allocated GPU memory and tokens/sec on the same workload before choosing a
chunk size. GPU savings have not been benchmarked in the current environment.

Your existing training command and checkpoint format continue to work. Restart
the training process to load the updated code and configuration. This is a memory
optimization that preserves the objective, not evidence of improved answer
accuracy; evaluate held-out loss and task responses to measure model quality.
