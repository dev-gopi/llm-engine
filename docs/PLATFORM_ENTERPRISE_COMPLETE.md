# Platform Workers + Enterprise Control Plane

Implemented in main(16) on top of main(15), preserving existing distributed-state and multimodal behavior.

## Platform workers
- Durable SQLite and Redis worker discovery/state transitions for Vector Stores, Batches and Fine-tuning Jobs.
- Vector ingestion parses text/Markdown/JSON/JSONL/CSV/YAML/HTML and optional PDF/DOCX, chunks content, embeds it and persists a searchable vector index.
- Batch worker validates JSONL line-by-line, enforces the declared endpoint, executes through an injectable async executor, writes OpenAI-style output/error JSONL files and persists request counts.
- Fine-tuning worker materializes an SFT training config from the selected fine-tuning profile, runs `scripts/train.py`, records logs/errors and registers a resulting fine-tuned model identifier.
- `scripts/run_platform_workers.py` supports one-shot CronJob or long-running worker operation and works with local SQLite or Redis shared state.

## Enterprise
- Tenant feature entitlements and usage metering.
- Moderation pipeline with executable local rules and optional classifier fusion.
- Data residency, legal hold, retention enforcement and compliance reporting.
- SCIM-style user/group directory and local/external policy evaluation.
- Tamper-evident hash-chained and HMAC-signed audit export.
- Artifact manifest/SBOM hashing, signing and admission verification.
- Environment, AWS Secrets Manager and Vault secret retrieval; AWS KMS, local HMAC and optional PKCS#11 HSM signing.
- Hardened verifier execution wrapper using bubblewrap/firejail when installed, with a sanitized environment and timeout.
- Kubernetes Deployment/HPA generation, SLO promotion/rollback decision logic, replica registry/routing and multi-region failover selection.
- Backup/restore primitives and multi-destination tenant webhook subscriptions.

## Qualification boundary
All paths above are implemented and unit-tested at source level. External qualification still depends on the operator environment: Redis/Redis Stack, cloud KMS/Secrets Manager, Vault, PKCS#11 HSM, Kubernetes, multi-region networking/storage, real model training hardware, and production moderation models cannot be certified by local unit tests alone.
