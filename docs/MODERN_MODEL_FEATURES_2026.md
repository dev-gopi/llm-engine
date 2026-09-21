# Modern Model Feature Research — September 2026

This note records the external model/runtime patterns used to guide the optional
modernization work in this repository. It is **not** a claim that the active
81.3M Gopi checkpoint matches the quality, scale, kernels, or context length of
these reference models.

## Verified external references

### Qwen3.6-27B configuration

The published Qwen3.6-27B configuration uses a repeating three-linear / one-full
attention schedule across 64 layers, 24 query heads, 4 KV heads, a 262,144-token
maximum position setting, and one MTP hidden layer. The production linear layer
is architecture-specific and is more sophisticated than the reference
ELU+1 recurrent linear attention implemented in this repository.

Source: <https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json>

### GLM-5.3 runtime controls

The GLM-5.3 model card documents explicit `reasoning_effort` controls and
serving through multiple optimized runtimes including SGLang, vLLM,
Transformers, KTransformers, and others. This reinforces the design choice to
keep reasoning budgets and runtime adapters as explicit API/runtime contracts
instead of baking one serving engine into the model implementation.

Source: <https://huggingface.co/zai-org/GLM-5.3/blob/main/README.md>

## Features adopted in Gopi without changing the default checkpoint

| Feature | Repository implementation | Compatibility policy |
| --- | --- | --- |
| Hybrid attention schedule | `attention_layer_pattern` plus `CausalLinearAttention` | Opt-in; `configs/model.gpu.yaml` is unchanged |
| Fixed-state linear decode | recurrent key/key-value sums | Opt-in; dense checkpoints retain projection shapes |
| Long-context research profile | `configs/model.hybrid.gpu.yaml` | Requires checkpoint-backed CTX validation before capability claims |
| Sparse MoE balancing | router auxiliary loss + entropy/load diagnostics | Zero weight by default; ordinary dense training unchanged |
| MTP auxiliary training | existing MTP heads/trainer integration | Opt-in via MTP profile |
| Q1_0 research packing | self-describing 1.125-bit packed safetensors | Engine-native research format; not claimed GGUF-binary compatible |
| GGUF runtime support | delegated OpenAI-compatible backend + `serve_gguf.py` | Uses specialized external runtime for GGUF/custom kernels |
| Low-bit deployment planning | BF16/INT8/INT4/Q1 weights and BF16/INT8/INT4 KV matrix | Analytical estimate only |
| Reasoning effort | `none` / `low` / `medium` / `high` API/runtime control | Existing capability now exposed in the playground |
| Architecture reporting | API capabilities + training-report architecture/memory panels | Derived from active config/runtime evidence |

## Deliberately not claimed

The repository does not claim 1-bit quality retention,
Qwen3.6-compatible linear-convolution kernels, 262K trained-context quality,
DSpark-equivalent speculative speedup, GLM-scale sparse attention, native
GPTQ/AWQ/GGUF export, or frontier-model benchmark quality. Those require
specific training, kernels, checkpoints, hardware measurements, and release
benchmarks. The relevant task-registry items remain open where that evidence is
not present.
