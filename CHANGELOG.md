# Changelog

Notable user-visible changes are recorded here. The project currently follows
the version in `pyproject.toml`; it has not yet declared a stable public API.

## Unreleased

- Added dedicated v2 training, usage, capabilities, troubleshooting, dataset,
  API, and deployment documentation.
- Added an OpenAI-compatible text chat interface for third-party UIs.
- Added Swagger, ReDoc, browser UI, health, readiness, metrics, authentication,
  rate limiting, and model reload support to local serving.
- Added a non-root Docker image, CPU/GPU Compose deployment, TLS Nginx reverse
  proxy, trusted-host checks, security headers, protected metrics, and
  production documentation controls.
- Run Compose serving with the host UID/GID for read-only private model mounts,
  and separate the public proxy edge network from the internal API network.
- Added domain-specific pretraining and fine-tuning evaluation workflows.
- Added packed token shards, mixed-precision evaluation, consistent EMA,
  nonblocking data transfers, persistent workers, GQA, and KV-cache generation
  optimizations.
- Added safe tokenizer vocabulary extension with compatibility validation and
  embedding/output-layer resizing.
- Added dataset preparation, governance checks, multilingual/code data support,
  and DPO preference training.
- Expanded training logs with progress, ETA, validation baseline, GPU memory,
  throughput, gradient health, and non-finite update tracking.

## 0.1.0

- Initial configuration-driven GPT model, tokenizer, training, evaluation,
  generation, export, and serving implementation.
