# Changelog

## Unreleased

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
