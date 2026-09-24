# Changelog

## Unreleased

- Added browser-playground support for typed image, audio, and video Responses inputs; audio/video files are uploaded as authenticated media assets before generation.
- Renamed versioned Omni router helpers and internal media settings to descriptive speech, video, and multimodal names.
- Fixed Omni video-understanding requests to preserve standard serving error statuses (for example, 503 when the generation backend is unavailable) instead of incorrectly returning HTTP 500.
- Fixed serving shutdown to cancel active continuous streams and release token-step decode state, preventing hung shutdowns and leaked KV-page allocations. Failed external-backend startup now also closes its HTTP connection pool.
- Fixed model sizing and training-compute estimates for optional multi-token-prediction heads, and reject invalid boolean or non-integral MTP-head counts in model configs.
- Fixed cached decoding with a current-token-only attention mask: automatic position IDs now continue from the KV-cache offset instead of restarting at zero. This preserves learned, sinusoidal, and RoPE position semantics during generation.
- Fixed `evaluate_benchmarks.py --long-context-lengths` to load the model configuration before validating requested context lengths, returning the intended actionable CLI error instead of raising a `NameError`.
- Moved the project data-pipeline package to `local_dataset` so it no longer conflicts with the optional Hugging Face `datasets` dependency.
