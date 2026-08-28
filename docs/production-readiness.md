# Production readiness

This repository is a tested training and serving engine, but it is not yet a
cluster-scale production platform. Passing unit tests establishes local
correctness; it does not establish throughput, fault tolerance, model quality,
security compliance, or safe operation under hostile traffic.

## Safeguards implemented

- Tokenizer input is balanced by contributed UTF-8 bytes rather than source
  order, with only one active shard per configured source.
- Tokenizers, model checkpoints, and token-shard manifests carry compatibility
  fingerprints.
- Single-file checkpoints are atomic and distributed checkpoints preserve
  rank-local runtime state.
- Mixed precision uses FP32 loss arithmetic, clipping, dynamic FP16 scaling,
  nonfinite-update rejection, and BF16 capability checks.
- Public request schemas are bounded, generation concurrency has deadlines,
  stream queues have backpressure, and disconnected streams cancel backend
  work.
- Model startup fails closed on vocabulary mismatches and does not select an
  arbitrary recently modified checkpoint.
- MCP subprocesses receive a minimal environment by default and tool catalogs
  remain allowlisted.
- Session and rate-limit state have expiry cleanup instead of unbounded key
  growth.

## Blocking gaps for large-scale deployment

1. **No true continuous batching or paged-attention execution.** The stream
   scheduler multiplexes independent decode loops. `DynamicBatcher` is not a
   token-level scheduler, and the page allocator is currently used only for
   cached prefixes. A high-throughput service needs one decode scheduler that
   batches active sequences every token step.
2. **Large models are fully materialized before sharding.** Tensor parallelism
   loads a complete checkpoint on every rank and then slices it. FSDP model
   construction also begins from ordinary parameter allocation. Multi-billion
   parameter deployment needs meta-device initialization and directly sharded
   checkpoint loading.
3. **No elastic training control plane.** There is no automatic worker restart,
   preemption handler, object-store checkpoint publication, checkpoint
   retention policy, or corruption checksum. Distributed checkpoint directories
   are not transactionally published as a complete unit.
4. **Data preparation is single-node.** Tokenization, filtering, deduplication,
   and shard construction are not distributed and do not persist resumable
   progress. Corpus-scale exact/near deduplication needs a distributed index.
5. **Evaluation is insufficient for release gates.** Keyword benchmark scoring
   and validation loss must be supplemented with held-out contamination-safe
   evaluations for reasoning, coding, multilingual ability, safety, factuality,
   latency, and regression testing.
6. **Serving operations are basic.** `/metrics` returns JSON rather than
   Prometheus metrics; there is no tracing, structured request audit stream,
   autoscaling signal, circuit breaker, load shedding by token budget, or
   multi-region routing.
7. **Security is application-level only.** A static bearer key, local TLS
   termination assumptions, and regex safety filters are not substitutes for
   an identity provider, RBAC, secrets manager, network sandbox, egress policy,
   malware scanning, and production moderation.

## Required release gates

- Use immutable, explicitly configured model/tokenizer/checkpoint artifact IDs.
- Set dataset governance to `error` and archive reviewed manifests.
- Run distributed collective, checkpoint-resume, and numerical soak tests on
  the exact target hardware.
- Load-test prompt-length distributions, cancellation, queue saturation, and
  GPU out-of-memory recovery.
- Benchmark quality before and after quantization or parallelization.
- Terminate TLS at a trusted proxy, configure authentication and rate limits,
  and keep MCP environment inheritance disabled.
- Deploy canaries, define rollback criteria, and verify backup restoration.
