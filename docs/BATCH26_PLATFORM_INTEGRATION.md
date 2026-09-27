# Batch 26 — production platform integration

This batch connects previously isolated serving primitives and adds persistent OpenAI-compatible control-plane APIs.

## Implemented

1. Server-configured Redis semantic cache construction.
2. Cross-replica Redis semantic nearest-neighbor lookup with policy isolation.
3. Distributed semantic caching for streaming responses.
4. Tenant semantic-cache quota enforcement.
5. Distributed cache warming endpoint.
6. Selective distributed cache purge endpoint.
7. Negative-cache administrative clearing.
8. Hot-swappable LoRA publish/list/activate/deactivate/remove lifecycle API.
9. EBNF grammar constraints in `/v1/generate`.
10. EBNF grammar constraints in `/v1/chat/completions`.
11. OIDC/JWT bearer authentication with JWKS or HS256 development mode.
12. Role-based admin authorization.
13. Authenticated tenant/user binding into generation/cache identity.
14. Per-tenant request quotas.
15. Optional OpenTelemetry tracing/OTLP export.
16. Persistent OpenAI-compatible Files API.
17. Persistent OpenAI-compatible Vector Stores lifecycle API.
18. Persistent OpenAI-compatible Batch lifecycle API.
19. Persistent OpenAI-compatible Fine-tuning Jobs lifecycle API.
20. Standard `/v1/images/generations` compatibility bridge.
21. Reusable cross-service idempotency-key store used by multiple mutation APIs.
22. HMAC-signed webhook delivery with bounded retries and dead-letter storage.

## Important boundaries

The Vector Stores API now has lifecycle/file-association persistence, but a scalable ingestion/chunk/embed/search worker remains separate work. Batch and Fine-tuning Jobs now have persistent OpenAI-compatible control-plane lifecycle records, but execution workers/orchestrators remain separate work. Redis semantic lookup is distributed and functional but currently uses a bounded candidate scan; an ANN/vector-index backend remains a scale optimization.

## Security defaults

OIDC is disabled unless configured. Existing API-key behavior is preserved. Authenticated identity overrides client-supplied tenant/user fields for generation. Admin endpoints accept either the dedicated admin secret or an OIDC principal with the `admin` role.
