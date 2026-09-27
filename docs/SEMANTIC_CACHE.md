# Semantic response cache

`llm-engine` includes an opt-in, conservative two-tier response cache:

1. exact request identity lookup;
2. semantic prompt lookup using normalized embeddings;
3. normal generation on a miss.

The feature is disabled by default. Enable it only after validating the embedding model and similarity threshold on representative traffic.

## Safety-first eligibility

By default a request is cacheable only when generation is deterministic (`temperature: 0`) and it does not use session memory, legacy or OpenAI tools, MCP, web search, RAG, text attachments, or reasoning output. Those inputs are dynamic or private and are intentionally bypassed rather than risking stale or cross-context reuse.

Every semantic candidate must have the same non-prompt generation policy. The policy fingerprint includes the request schema and generation controls; the namespace is deployment-controlled and should change whenever model/checkpoint, tokenizer, system-prompt policy, or other serving semantics change.

## Configuration

```yaml
serving:
  semantic_cache_enabled: false
  semantic_cache_capacity: 512
  semantic_cache_similarity_threshold: 0.985
  semantic_cache_ttl_seconds: 3600
  semantic_cache_namespace: gopi-v1
  semantic_cache_store_path: data/cache/semantic-responses.sqlite
  semantic_cache_allow_nondeterministic: false
```

Environment variables:

- `GOPI_SEMANTIC_CACHE_ENABLED`
- `GOPI_SEMANTIC_CACHE_CAPACITY`
- `GOPI_SEMANTIC_CACHE_THRESHOLD`
- `GOPI_SEMANTIC_CACHE_TTL_SECONDS`
- `GOPI_SEMANTIC_CACHE_NAMESPACE`
- `GOPI_SEMANTIC_CACHE_STORE`
- `GOPI_SEMANTIC_CACHE_ALLOW_NONDETERMINISTIC`

SQLite persistence is optional. Without `semantic_cache_store_path`, the cache is in-process only.

## Operations

`GET /metrics` exposes `semantic_cache_*` counters including exact hits, semantic hits, misses, bypasses, stores, evictions, expiration, entry count, and hit rate.

Authenticated admin endpoints:

- `GET /admin/cache/semantic` — inspect cache state and counters.
- `DELETE /admin/cache/semantic` — purge the active namespace.

## Current intentional limits

This implementation does not yet provide Redis/vector-database distribution, per-tenant quotas, per-request `no-store`, stale-while-revalidate, negative/error caching, cache warming, semantic-hit debug metadata, or offline false-hit evaluation tooling. RAG/tool/web/session requests are bypassed instead of being cached with freshness fingerprints.
