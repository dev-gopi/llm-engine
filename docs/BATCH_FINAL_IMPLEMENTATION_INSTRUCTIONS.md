# Final A-to-Z implementation instruction

1. Treat the Batch-23 patched repository as immutable baseline behavior.
2. Do not replace working APIs when an additive adapter or compatibility layer is sufficient.
3. For every remaining feature, first inspect the existing implementation and tests.
4. Add the smallest production-safe implementation that preserves current defaults.
5. External runtimes (CUDA, Redis, vLLM, TensorRT-LLM, FlashInfer, FlashAttention, multimodal providers) must be capability-gated and must fail explicitly when unavailable.
6. Never label a portable approximation as a vendor-native format.
7. Run targeted regression tests after each subsystem change.
8. Run compileall before packaging.
9. Do not remove an audit item unless its acceptance criteria are actually met.
10. Keep production qualification distinct from code-level implementation.
