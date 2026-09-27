# Missing-feature implementation status

This file tracks the incremental hardening work applied after `llm-engine-main(6).zip`.

## Iteration 1 — semantic response cache

Implemented:
- opt-in exact-response cache tier;
- semantic similarity tier through the existing pluggable `EmbeddingService`;
- canonical request/policy fingerprints;
- deterministic-only safe default eligibility;
- automatic bypass for sessions, tools, MCP, RAG, web search, attachments, and reasoning;
- configurable similarity threshold, TTL, capacity, namespace and nondeterministic opt-in;
- in-memory LRU plus optional SQLite persistence;
- cache namespace invalidation/versioning;
- streaming capture and replay;
- in-process single-flight protection for identical requests;
- semantic-cache metrics in `/metrics`;
- authenticated cache status/purge endpoints;
- documentation and environment/config examples;
- dedicated unit/integration tests.

Verification:
- `tests/test_semantic_cache.py`, `tests/test_serving.py`, and `tests/test_serving_orchestration.py`: 62 passed.
- Full suite collection is blocked in the audit container because `pyarrow` is not installed. The dependency is already declared in `pyproject.toml`.
- A broader run excluding the five pyarrow-dependent modules progressed past 64% with no failures before the execution timeout.

Remaining semantic-cache production items and all other platform gaps are tracked in `docs/MISSING_FEATURES_AUDIT.md`.

No claim of mathematical or production "100% accuracy" is made: correctness is established incrementally through tests, compatibility checks, and runtime validation.

## Iteration 2 — distributed preference training + IPO + SFT CLI

Implemented:
- two-process and multi-process DDP support in `scripts/train_dpo.py` via the existing `DistributedTrainer`;
- rank-aware preference dataloaders using `DistributedSampler`;
- distributed aggregation of train/validation preference metrics;
- rank-0-only checkpoint writes and final JSON output;
- method-aware checkpoint metadata and resume validation;
- IPO loss/training support alongside DPO (`method: ipo` or `--method ipo`);
- CPU/GPU IPO profiles (`configs/ipo.cpu.yaml`, `configs/ipo.gpu.yaml`);
- `scripts/train_sft.py` convenience entry point that delegates to the production SFT path in `scripts/train.py`, preserving DDP/FSDP, resume, PEFT, reporting, and checkpoint behavior;
- alignment-method config now advertises only operational pairwise methods under `methods`, with ORPO/KTO/GRPO/PPO kept in `planned_methods` rather than falsely presenting them as implemented.

Verification:
- DPO/IPO workflow, alignment, SFT wrapper, and distributed-loader tests: passed.
- real two-process CPU/Gloo DDP smoke test: passed and verified synchronized policy parameters.
- targeted training regression suite: **82 passed, 1 skipped**.

Current limitation:
- distributed preference training is DDP. FSDP preference training still needs sharded checkpoint/reference-model handling before it can be enabled safely.

## Iteration 3 — ORPO + generation/embedding/workspace API parity

Implemented:
- ORPO pairwise preference objective with reference-free policy training;
- `--method orpo` plus CPU/GPU ORPO profiles;
- OpenAI-style presence and frequency penalties in native single, batched, batched-stream, and streaming generation paths;
- presence/frequency forwarding for the external OpenAI-compatible backend;
- embeddings `encoding_format=base64` using little-endian float32 bytes;
- requested embedding dimensions from 1 through the encoder native dimension;
- workspace patch creation/deletion through reviewed unified diffs while preserving root-boundary validation and `git apply --check` safety;
- alignment method registry updated so ORPO is operational rather than planned.

Verification:
- targeted feature suite: **82 passed**;
- broader serving/training/generation/preference regression suite: **219 passed, 1 skipped**;
- `python -m compileall -q src scripts`: passed.

Current limitation:
- FSDP preference training remains intentionally disabled until CUDA sharded checkpoint/optimizer/reference-model state handling can be exercised end-to-end. This iteration does not pretend that a CPU-only unit test proves FSDP production correctness.

## Iteration 4 — KTO + chat logprobs + multi-choice completions

Implemented:
- KTO binary-feedback training with native `prompt` + `completion` + `desirable`/`label` records;
- migration support that expands existing chosen/rejected pairs into desirable/undesirable KTO examples;
- KTO loss with policy/reference reward ratios, detached non-negative KL baseline, configurable class weights, checkpoint method validation, DDP-compatible loaders/metric reduction;
- CPU/GPU KTO profiles and alignment registry promotion from planned to operational;
- OpenAI-style chat `logprobs` and `top_logprobs` request validation and response payloads;
- token-level selected logprob plus configurable top alternatives in native single-stream and continuous-batched decode paths;
- logprob forwarding/parsing through the external OpenAI-compatible backend;
- chat `n` multiple-choice responses for non-streaming and streaming requests, with deterministic seed offsets when a seed is supplied;
- explicit rejection of `n > 1` with `session_id` so multiple alternatives cannot silently corrupt a single conversation history;
- semantic-cache preservation of token logprob metadata.

Verification:
- focused KTO/generation/chat tests: **76 passed**;
- serving/API/conformance/external/semantic-cache/preference suite: **140 passed**;
- broader regression selection across generation, DPO/KTO, distributed DPO smoke, serving/orchestration, API/conformance, configs, training integration/system/accounting and workspace: **295 passed, 2 skipped**;
- `python -m compileall -q src scripts`: passed.

Current limitations:
- KTO is supported with DDP but preference-training FSDP remains intentionally disabled pending CUDA sharded checkpoint/reference-model tests;
- multi-choice requests with session memory are rejected by design because choosing which alternative becomes history is an application decision.

## Iteration 5 — reward-model training + offline GRPO

Implemented:
- scalar decoder-backbone reward model with a trainable reward head on the final non-padding hidden state;
- opt-in hidden-state return from `MiniGPT.forward` without changing the default logits-only behavior;
- Bradley-Terry/logistic chosen-vs-rejected reward-model objective with accuracy/margin/chosen/rejected reward metrics;
- `scripts/train_reward_model.py` with initialization from an SFT/policy checkpoint, resume support, checkpoint metadata, early stopping, DDP-compatible dataloading/reductions and CPU/GPU profiles;
- offline/pre-scored GRPO dataset contract using `prompt`, `completions`, and per-completion `rewards` while preserving prompt groups;
- clipped group-relative policy objective with within-group normalized advantages, frozen old-policy/reference baselines, KL regularization and clip/KL/reward metrics;
- `scripts/train_grpo.py` with DDP-compatible loading/reductions, exact old-policy checkpoint metadata on resume, validation/best checkpoints and CPU/GPU profiles;
- alignment registry now distinguishes operational `grpo_offline` from still-planned `grpo_online` and PPO.

Verification:
- focused reward-model/GRPO plus existing DPO/generation compatibility tests: **84 passed**;
- broader serving/API/conformance/model/post-training regression selection: **296 passed**;
- `python -m compileall -q src scripts`: passed.

Current limitations:
- GRPO is currently the offline/pre-scored variant. Online rollout generation, reward/verifier/tool execution during rollouts, replay/rollout workers and iterative old-policy refresh are still missing;
- reward-model training is operational, but PPO-style end-to-end RLHF and reward-model serving/calibration are separate remaining work;
- reward-model and GRPO distributed execution currently use DDP, not FSDP.


## Iteration 6 — reward calibration/scoring + online GRPO core

Implemented:
- portable reward calibration metadata with scalar reward mean/std normalization, clipping and deterministic pairwise temperature fitting;
- held-out reward-model calibration/evaluation CLI with pairwise accuracy, tie-rate, NLL and Brier metrics;
- batch reward scoring CLI for `prompt` + `completion` JSONL with optional normalized rewards;
- reusable `RewardScorer` API for post-training and rollout code;
- live policy rollout generation for GRPO using the production `Generator`;
- reward-model scoring during rollout generation plus an exact-match verifier signal that can be mixed by configurable weights;
- per-prompt rollout fault isolation, persisted failure logs and atomic rollout JSONL writes;
- iterative old-policy refresh before each online GRPO iteration;
- bounded replay-buffer support with persistent JSONL recovery;
- resumable single-process online GRPO checkpoints carrying the completed iteration, reference/reward/calibration provenance and replay location;
- in-memory GRPO loader support so generated rollout groups flow directly into the tested GRPO objective;
- CPU/GPU online-GRPO profiles and alignment registry promotion of `grpo_online` to operational.

Verification:
- reward calibration/scoring and online-rollout unit tests pass;
- a real tiny CPU smoke run successfully calibrated a reward model, generated a live GRPO group, trained one online GRPO update and wrote a resumable checkpoint;
- broader regression results are recorded in `docs/MISSING_FEATURES_AUDIT.md`.

Current limitations:
- online GRPO is intentionally single-process. Distributed/asynchronous rollout workers, rollout sharding/aggregation and cross-node recovery remain separate work;
- the built-in verifier signal is exact-match. Tool execution, sandboxed code tests and pluggable verifier farms are not yet wired into the online rollout loop;
- reward scoring/calibration is available as a reusable Python API and CLI, but a separately authenticated production reward-scoring HTTP service is still missing;
- PPO/value-model RLHF and FSDP post-training remain separate work.

## Current iteration — FSDP preference/reward/offline-GRPO wiring

Implemented on top of the PPO + distributed-online-GRPO codebase:
- DPO/IPO/ORPO/KTO CLIs now accept `distributed_strategy: fsdp` and `fsdp_hybrid` under multi-rank CUDA launches;
- reward-model training now supports FSDP/hybrid-FSDP policy wrapping and exact sharded resume;
- offline GRPO policy training now supports FSDP/hybrid-FSDP while retaining full per-rank reference/old-policy evaluators;
- FSDP post-training checkpoint format v2 stores sharded model + optimizer state together with scheduler, scaler, trainer metadata and world-size manifest validation;
- exact resume rejects accidental single-file/FSDP format mixing rather than silently losing optimizer state;
- DDP/single-process checkpoint paths remain backward compatible;
- dedicated FSDP example configs were added for DPO, reward-model and offline GRPO.

Verification boundary:
- source/CLI wiring, checkpoint manifest behavior, compile checks and non-CUDA regressions are covered;
- real multi-GPU CUDA FSDP execution remains target-hardware qualification work;
- online GRPO/PPO FSDP, FSDP generation evaluation, and elastic world-size resharding remain separate tasks.
