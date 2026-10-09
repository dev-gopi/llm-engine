# Changelog

## Unreleased
- Fixed native token-step serving with tenant LoRA adapters: streams snapshot
  adapter state at admission, decode compatible snapshots in one batch, and
  isolate distinct adapters in separate lock-protected model calls.
- Fixed paged-KV allocator accounting and failed-admission cleanup, and exposed
  the existing optional INT8 paged-KV storage through
  `serving.paged_kv_quantization`.
- Corrected the active-model capability summary to its configured 1,024-token,
  approximately 81.3M-parameter architecture.
- Sized the active 4 GB serving profile for four full-context paged-KV streams,
  bounded long-prompt prefill to 256 tokens per pass, and capped retained
  prefixes to its configured page-pool headroom.
- Implemented native executable code and CLI entry points across the production backlog:
  - Added `src/runtime/moe_execution.py` enabling real forward execution and memory estimation for 100B sparse MoE and hybrid MoE 7B configurations, with meta-device execution support in `MoEFeedForward`.
  - Added native MiniGPT quantization CLIs `scripts/quantize_gptq.py` and `scripts/quantize_awq.py`, with group-wise quantization, packing, and deployment manifest generation integrated into `scripts/export.py`.
  - Implemented tenant-isolated LoRA concurrency safety in `src/serving/backend.py` with scoped adapter locking to prevent cross-tenant race conditions.
  - Added `TensorRTLLMServingBackend` in `src/serving/backend.py` with serving reload and environment configuration wiring.
  - Added `CoquiXTTSVoiceCloningProvider`, `VoiceCloneProfile`, and `VoiceCloneTrainer` in `src/omni_platform/voice_cloning.py` for voice cloning training and inference runtime.
  - Implemented multimodal endpoints in `RemoteJSONProvider` (`src/omni_platform/provider_adapters.py`) and PCM-to-WAV streaming chunking in `src/omni_platform/audio_responses.py`.
- Added a guarded native AutoAWQ artifact loader. Verified the existing main
  export CLI's native and llama.cpp-compatible GGUF paths.
- Added a guarded native AutoGPTQ artifact loader alongside the existing GPTQ
  conversion bridge. Both require a compatible optional GPTQ runtime.
- Added optional torchao-backed CUDA INT8/INT4 weight-only inference paths.
  They require BF16, a CUDA runtime, and a compatible torchao installation.
- Added a production-feature checklist that records code-level completion and
  target-runtime qualification separately for the active implementation backlog.
- Fixed the production-runtime qualification CLI when invoked directly from the
  repository: it no longer lets `scripts/tokenize.py` shadow Python's standard
  library `tokenize` module.
- Added optional step-limited language-model training through YAML `max_steps`
  or `scripts/train.py --max-steps`. The limit counts optimizer updates,
  preserves mid-epoch resume state, and caps the fresh-run learning-rate plan.
- Added optional micro-batch and supervised-token limits, wall-clock checkpoint
  intervals, and automatic resume from an existing configured latest checkpoint.
- Fixed training-report checkpoint charts when several checkpoint artifacts are
  saved at the same optimizer step. Each checkpoint kind now renders as its own
  series, preventing misleading vertical line spikes.
- Fixed eight core-model correctness issues: supplied KV caches are honored
  when gradient checkpointing is enabled; paged decoding respects each request's
  sliding window; YaRN applies frequency scaling once; top-1 MoE routers receive
  prediction-loss gradients; MoE capacity limits split expert work instead of
  dropping competing tokens; strict causal checkpoint loading rejects missing
  or unexpected core weights in MTP models; BF16-model memory estimates account
  for FP32 linear-attention state; and vocabulary resizing preserves frozen
  output-head weights and biases. YaRN and top-1/capacity-limited MoE predictions
  intentionally change while checkpoint tensor layouts remain compatible.
- Tenant-scoped media assets now prevent cross-tenant reads, deletion, and Omni processing; Omni providers are cached and synchronous media work is offloaded from the event loop. Voice cloning now requires a short-lived server-verified consent assertion bound to the tenant and reference asset, and realtime cancellation cancels active generation tasks.
- Unified Omni routes with the configured serving authentication policy, so `authentication_enabled: false` now consistently disables API-key enforcement across all protected transports.

- Fixed OIDC-only realtime WebSocket authentication and added the explicit `authentication_enabled` serving switch (`GOPI_AUTHENTICATION_ENABLED`) for controlled local enablement or disablement.

- Configured the checked-out local `.env` for unauthenticated loopback use and
  removed its public tunnel origin. API and admin keys remain explicit opt-in
  requirements before any non-local exposure.

- Fixed Gloo distributed workers to remain on CPU even when CUDA is visible,
  preventing CPU smoke tests from allocating GPU contexts. Updated the OIDC
  HS256 test fixture to use a standards-compliant signing-key length.

- Batch 21: enabled local recursive `$ref` and `$dynamicRef` JSON Schema validation for structured outputs using the Draft 2020-12 resolver, with recursive tree and dynamic-anchor regression coverage; external URI retrieval remains disabled.

- Modernized the `/ui/` text-generation playground with a refined chat canvas, prompt suggestions, and a focused generation composer while retaining streaming, multimodal attachments, tools, and API-mode controls.
- Made the text-generation playground Stop action permanently visible and generation-aware: it becomes active while a response streams and cancels the current request safely.
- Added unified system reporting and interactive web dashboard for inference, serving, chat session memory, and RAG knowledge retrieval. Includes `src/inference/reporting.py`, `src/serving/report_router.py`, `scripts/build_report.py`, and `scripts/serve_report.py` with standalone offline HTML export and live interactive generation and RAG search playgrounds.
- Made the standalone system report load `system_report.json` by default, added an explicit JSON link, and included a sample report fallback for offline viewing.
- Updated the training report UI to match the system report's operations dashboard style, with a live status badge, elevated controls, and categorized navigation.
- Refined the training, standalone system, and served operations dashboards with modern responsive surfaces, accessible focus states, improved controls, and clearer generation playground and response presentation.
- Added a responsive generation-response modal to the standalone and served system reports, including benchmark and live-playground results, performance metadata, full prompt/output viewing, Escape-to-close, focus restoration, and one-click response copying.
- Improved the live training report: validation is now shown while it runs, with per-batch ETA when available or a clearly labelled historical-duration estimate; optimizer-step, log-window, checkpoint, and validation timings now use separate charts so their scales do not obscure one another.
- Fixed runtime timing contamination in training and validation report generation: reset log interval timing after initial/mid-epoch validations and checkpoint saves, fixed `Trainer.promotion_metrics()` calling `tokens_per_second` property as a function, fixed validation progress ETA formatting when target batches are unknown, and added training and validation start and end timestamps to report progress analysis.
- Added a coding-mode tool workflow that guides compatible clients through inspect, analyze, patch, and verify stages.
- Upgraded MCP client negotiation to fall back across supported legacy protocol versions and added `streamable_http` server wiring for serving and the MCP CLI; tool allowlists remain mandatory.
- Added RAG support for image metadata (PNG/JPEG/WebP/GIF/BMP/TIFF) and safely bounded ZIP archives containing text, code, and configuration files.
- Extended local RAG ingestion to PDF, DOCX, XLSX/XLSM, and common source-code/configuration files. Excel workbooks are read-only with cached values; all retrieved content remains untrusted.
- Made Pillow a standard runtime dependency because the supported image-data
  pipeline requires it, and fixed direct execution of affected audit and
  benchmark CLIs by applying the shared script import-path bootstrap.
- Added an explicit `validation_lr_adaptation_enabled` switch and persisted plateau-recovery spacing across checkpoint resume, so validation-driven LR adaptation can be safely enabled or disabled per profile.
- Enabled validation-driven learning-rate plateau recovery in the primary CPU
  and GPU pretraining and fine-tuning profiles. Training now retains the best
  checkpoint, lowers learning rate on sustained validation regressions, then
  early-stops; it never changes checkpoint-incompatible model architecture.
- Added browser-playground support for typed image, audio, and video Responses inputs; audio/video files are uploaded as authenticated media assets before generation.
- Renamed versioned Omni router helpers and internal media settings to descriptive speech, video, and multimodal names.
- Fixed Omni video-understanding requests to preserve standard serving error statuses (for example, 503 when the generation backend is unavailable) instead of incorrectly returning HTTP 500.
- Fixed serving shutdown to cancel active continuous streams and release token-step decode state, preventing hung shutdowns and leaked KV-page allocations. Failed external-backend startup now also closes its HTTP connection pool.
- Fixed model sizing and training-compute estimates for optional multi-token-prediction heads, and reject invalid boolean or non-integral MTP-head counts in model configs.
- Fixed cached decoding with a current-token-only attention mask: automatic position IDs now continue from the KV-cache offset instead of restarting at zero. This preserves learned, sinusoidal, and RoPE position semantics during generation.
- Fixed `evaluate_benchmarks.py --long-context-lengths` to load the model configuration before validating requested context lengths, returning the intended actionable CLI error instead of raising a `NameError`.
- Moved the project data-pipeline package to `local_dataset` so it no longer conflicts with the optional Hugging Face `datasets` dependency.
