# Distributed training and inference completion layer

This batch adds production-oriented distributed and inference primitives while preserving the existing DDP/FSDP and serving paths.

## Distributed training

- `training.parallelism.ParallelMesh` builds deterministic DP/PP/TP/EP/CP/SP process groups.
- `training.deepspeed` now supports ZeRO stages 0-3, CPU optimizer/parameter offload, native DeepSpeed save/resume helpers, and precision-aware configuration.
- `training.pipeline_parallel.DistributedPipelineStage` implements point-to-point activation and gradient transport for true cross-rank pipeline stages with static activation shapes.
- `training.expert_parallel` adds variable-size all-to-all token routing to expert owners and the reverse return path.
- `training.context_parallel` adds reduce-scatter, sequence-parallel linear execution and shard-aware causal masks.

These primitives are additive. Existing DDP/FSDP configs continue to use the previous code paths unless a new parallel runtime is explicitly selected by an integrator.

## Inference

- MiniGPT tensor parallelism now supports Sparse-MoE expert FFNs while keeping routers replicated.
- TensorRT-LLM has a real runner/tokenizer generation adapter instead of a hard-coded `NotImplementedError`.
- vLLM has an optional asynchronous engine adapter for native async streaming/event consumption.
- FlashInfer and FlashAttention adapters expose concrete attention execution entry points.
- Optional torchao CUDA INT4/INT8/FP8 quantization is exposed through `inference.gpu_quantization`.
- GPTQ/AWQ conversion supports native quantizable wrappers and explicit vendor converter hooks.
- `inference.backend_routing` adds circuit breaking, health-aware failover and tenant QoS queue primitives.

## Qualification boundary

Source-level implementation cannot prove vendor/runtime correctness on hardware that is not present. CUDA/ROCm GPUs, DeepSpeed, TensorRT-LLM, vLLM, FlashInfer, FlashAttention, torchao, GPTQ/AWQ runtimes and multi-node fabrics must still be exercised in the target deployment environment. No optional dependency is imported unless its backend is selected.
