"""Comprehensive reporting utilities for generation, serving, chat, and RAG."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch

from inference.context import ConversationMemory, format_system_prompt
from inference.generator import Generator
from inference.rag import SQLiteRagIndex
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from runtime.resource_planner import estimate_inference_memory
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


@dataclass(frozen=True)
class GenerationBenchmarkSample:
    prompt: str
    generated_text: str
    tokens_generated: int
    duration_seconds: float
    ttft_seconds: float
    tokens_per_second: float


@dataclass
class GenerationReport:
    model_name: str
    device: str
    samples_evaluated: int
    total_tokens_generated: int
    avg_tokens_per_second: float
    avg_ttft_seconds: float
    p95_ttft_seconds: float
    peak_vram_mb: float | None
    samples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ServingReport:
    status: str
    endpoint: str | None
    online: bool
    model_config_path: str
    checkpoint_path: str
    paged_kv_pages: int
    paged_kv_page_size: int
    prefix_cache_capacity: int
    requests_per_minute: int
    estimated_weight_memory_mb: float
    estimated_kv_cache_memory_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSessionSummary:
    session_id: str
    message_count: int
    last_updated: str | None


@dataclass
class ChatReport:
    total_sessions: int
    total_messages: int
    avg_messages_per_session: float
    database_path: str
    sessions_found: bool
    recent_sessions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RagReport:
    database_path: str
    index_exists: bool
    total_chunks: int
    sample_queries_tested: int
    avg_query_latency_ms: float
    sample_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SystemInferenceReport:
    timestamp: str
    generation: GenerationReport | None = None
    serving: ServingReport | None = None
    chat: ChatReport | None = None
    rag: RagReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "generation": asdict(self.generation) if self.generation else None,
            "serving": asdict(self.serving) if self.serving else None,
            "chat": asdict(self.chat) if self.chat else None,
            "rag": asdict(self.rag) if self.rag else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [
            f"# Engine Inference & Operations Report",
            f"**Generated at:** `{self.timestamp}`\n",
        ]
        if self.generation:
            lines.extend([
                "## 1. Generation & Inference Performance",
                f"- **Model:** `{self.generation.model_name}` ({self.generation.device})",
                f"- **Samples Evaluated:** {self.generation.samples_evaluated}",
                f"- **Tokens Generated:** {self.generation.total_tokens_generated}",
                f"- **Average Throughput:** `{self.generation.avg_tokens_per_second:.2f} tokens/s`",
                f"- **Time to First Token (TTFT avg / p95):** `{self.generation.avg_ttft_seconds * 1000:.1f} ms` / `{self.generation.p95_ttft_seconds * 1000:.1f} ms`",
            ])
            if self.generation.peak_vram_mb:
                lines.append(f"- **Peak VRAM:** `{self.generation.peak_vram_mb:.1f} MB`")
            lines.append("")

        if self.serving:
            lines.extend([
                "## 2. Serving & Deployment Configuration",
                f"- **Server Status:** `{'Online' if self.serving.online else 'Offline / Standalone'}`",
                f"- **Endpoint Checked:** `{self.serving.endpoint or 'N/A'}`",
                f"- **Paged KV Cache:** {self.serving.paged_kv_pages} pages (size {self.serving.paged_kv_page_size})",
                f"- **Prefix Cache Capacity:** {self.serving.prefix_cache_capacity} entries",
                f"- **Rate Limit:** `{self.serving.requests_per_minute} req/min`",
                f"- **Estimated Weight Memory:** `{self.serving.estimated_weight_memory_mb:.1f} MB`",
                f"- **Estimated KV Cache Memory:** `{self.serving.estimated_kv_cache_memory_mb:.1f} MB`",
                "",
            ])

        if self.chat:
            lines.extend([
                "## 3. Chat & Session Activity",
                f"- **Session Database:** `{self.chat.database_path}`",
                f"- **Total Recorded Sessions:** {self.chat.total_sessions}",
                f"- **Total Stored Messages:** {self.chat.total_messages}",
                f"- **Avg Messages per Session:** `{self.chat.avg_messages_per_session:.1f}`",
                "",
            ])

        if self.rag:
            lines.extend([
                "## 4. RAG Knowledge Base",
                f"- **Index Path:** `{self.rag.database_path}`",
                f"- **Total Indexed Chunks:** {self.rag.total_chunks:,}",
                f"- **Avg Search Latency:** `{self.rag.avg_query_latency_ms:.2f} ms`",
                f"- **Queries Benchmarked:** {self.rag.sample_queries_tested}",
                "",
            ])

        return "\n".join(lines)


def collect_generation_report(
    *,
    model_config_path: str | Path = "configs/model.gpu.yaml",
    inference_config_path: str | Path = "configs/inference.yaml",
    tokenizer_path: str | Path = "data/tokenizer-finetuning",
    checkpoint_path: str | Path = "checkpoints/finetuning/best.pt",
    prompts: Sequence[str] | None = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    device: str = "cpu",
) -> GenerationReport:
    """Run generation probes and collect latency, throughput, and memory stats."""
    model_cfg = load_yaml(model_config_path)
    inf_cfg = load_yaml(inference_config_path) if Path(inference_config_path).is_file() else {}
    bot_name = str(inf_cfg.get("bot_name", "Gopi"))

    tok = Tokenizer.load(tokenizer_path)
    model_cfg = adapt_config_to_tokenizer(model_cfg, tok)
    dev = resolve_device(device)

    model = MiniGPT.from_config(model_cfg, device="cpu")
    if Path(checkpoint_path).is_file():
        load_checkpoint(
            checkpoint_path,
            model,
            use_ema=True,
            restore_rng=False,
            **checkpoint_tokenizer_options(tok, allow_extension=False),
        )
    model.to(dev)
    model.eval()

    generator = Generator(model, tok, device=dev)
    test_prompts = list(prompts or [
        "Explain machine learning in simple terms.",
        "What is the function of an operating system?",
        "Write a brief python function to compute factorial.",
    ])

    samples: list[dict[str, Any]] = []
    ttfts: list[float] = []
    tps_list: list[float] = []
    total_tokens = 0

    if torch.cuda.is_available() and str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)

    for p in test_prompts:
        prompt_ids = tok.encode(p, add_bos=True)
        t_prefill_start = time.perf_counter()
        try:
            with torch.inference_mode():
                _ = generator._prefill(prompt_ids)
            ttft = max(time.perf_counter() - t_prefill_start, 1e-6)
        except Exception:
            ttft = 0.0

        start_time = time.perf_counter()
        res = generator.generate(p, max_tokens=max_tokens, temperature=temperature)
        duration = max(time.perf_counter() - start_time, 1e-6)
        token_count = len(res.token_ids)
        total_tokens += token_count

        if ttft <= 0.0:
            ttft = duration / max(token_count, 1)
        tps = token_count / duration

        ttfts.append(ttft)
        tps_list.append(tps)
        samples.append({
            "prompt": p,
            "generated_text": res.text.strip(),
            "prompt_tokens": res.prompt_tokens,
            "tokens": token_count,
            "duration_s": round(duration, 4),
            "ttft_s": round(ttft, 4),
            "tokens_per_second": round(tps, 2),
            "finish_reason": res.finish_reason,
        })

    peak_vram = None
    if torch.cuda.is_available() and str(dev).startswith("cuda"):
        peak_vram = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    sorted_ttfts = sorted(ttfts)
    p95_index = round((len(sorted_ttfts) - 1) * 0.95)

    return GenerationReport(
        model_name=bot_name,
        device=str(dev),
        samples_evaluated=len(test_prompts),
        total_tokens_generated=total_tokens,
        avg_tokens_per_second=round(statistics.mean(tps_list), 2) if tps_list else 0.0,
        avg_ttft_seconds=round(statistics.mean(ttfts), 4) if ttfts else 0.0,
        p95_ttft_seconds=round(sorted_ttfts[p95_index], 4) if sorted_ttfts else 0.0,
        peak_vram_mb=round(peak_vram, 2) if peak_vram is not None else None,
        samples=samples,
    )


def collect_serving_report(
    *,
    inference_config_path: str | Path = "configs/inference.yaml",
    endpoint: str = "http://localhost:8000",
) -> ServingReport:
    """Collect serving architecture parameters, memory estimates, and endpoint status."""
    inf_cfg = load_yaml(inference_config_path) if Path(inference_config_path).is_file() else {}
    serving_cfg = inf_cfg.get("serving", {})
    model_cfg_path = serving_cfg.get("model_config", "configs/model.gpu.yaml")
    checkpoint_path = serving_cfg.get("checkpoint_path", "checkpoints/finetuning/best.pt")

    # Sizing estimation
    weight_mb = 0.0
    kv_mb = 0.0
    if Path(model_cfg_path).is_file():
        m_cfg = load_yaml(model_cfg_path)
        try:
            mem = estimate_inference_memory(m_cfg, max_batch_size=int(serving_cfg.get("max_concurrency", 1)))
            weight_mb = mem.get("parameters_mb", 0.0)
            kv_mb = mem.get("kv_cache_mb", 0.0)
        except Exception:
            pass

    # Check live server endpoint
    online = False
    status_msg = "configured"
    try:
        req = Request(f"{endpoint.rstrip('/')}/readyz", headers={"User-Agent": "gopi-reporter"})
        with urlopen(req, timeout=2.0) as response:
            if response.status == 200:
                online = True
                status_msg = "ready"
    except (URLError, OSError, TimeoutError):
        online = False
        status_msg = "offline / idle"

    return ServingReport(
        status=status_msg,
        endpoint=endpoint,
        online=online,
        model_config_path=str(model_cfg_path),
        checkpoint_path=str(checkpoint_path),
        paged_kv_pages=int(serving_cfg.get("paged_kv_pages", 128)),
        paged_kv_page_size=int(serving_cfg.get("paged_kv_page_size", 16)),
        prefix_cache_capacity=int(serving_cfg.get("prefix_cache_capacity", 16)),
        requests_per_minute=int(serving_cfg.get("requests_per_minute", 60)),
        estimated_weight_memory_mb=round(weight_mb, 2),
        estimated_kv_cache_memory_mb=round(kv_mb, 2),
        metadata={
            "weight_dtype": serving_cfg.get("weight_dtype", "bfloat16"),
            "quantization": serving_cfg.get("quantization", "none"),
            "low_memory_loading": serving_cfg.get("low_memory_loading", True),
        },
    )


def collect_chat_report(
    *,
    database_path: str | Path = "data/cache/sessions.sqlite",
) -> ChatReport:
    """Analyze stored conversations, turn counts, and session history.

    Supports the production schema (id, messages JSON, updated float) and a
    legacy schema with explicit message_count / updated_at columns.
    """
    db_file = Path(database_path)
    if not db_file.is_file():
        return ChatReport(
            total_sessions=0,
            total_messages=0,
            avg_messages_per_session=0.0,
            database_path=str(db_file),
            sessions_found=False,
            recent_sessions=[],
        )

    try:
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if "sessions" not in tables:
                return ChatReport(
                    total_sessions=0,
                    total_messages=0,
                    avg_messages_per_session=0.0,
                    database_path=str(db_file),
                    sessions_found=False,
                    recent_sessions=[],
                )

            col_names = {r[1] for r in cursor.execute("PRAGMA table_info(sessions)").fetchall()}
            total_sessions = cursor.execute("SELECT count(*) FROM sessions").fetchone()[0]
            recent: list[dict[str, Any]] = []
            total_msgs = 0

            if "message_count" in col_names and "updated_at" in col_names:
                # Legacy / test schema: explicit message_count column
                total_msgs = sum(
                    r[0] for r in cursor.execute("SELECT message_count FROM sessions").fetchall()
                )
                rows = cursor.execute(
                    "SELECT session_id, message_count, updated_at FROM sessions "
                    "ORDER BY updated_at DESC LIMIT 10"
                ).fetchall()
                recent = [{"session_id": r[0], "message_count": r[1], "updated_at": r[2]} for r in rows]
            else:
                # Production schema: id (TEXT), messages (JSON list), updated (unix float)
                all_msgs = cursor.execute("SELECT messages FROM sessions").fetchall()
                for (msgs_json,) in all_msgs:
                    try:
                        msgs = json.loads(msgs_json or "[]")
                        total_msgs += len(msgs) if isinstance(msgs, list) else 0
                    except (json.JSONDecodeError, TypeError):
                        pass
                rows = cursor.execute(
                    "SELECT id, messages, updated FROM sessions ORDER BY updated DESC LIMIT 10"
                ).fetchall()
                for r in rows:
                    try:
                        n_msgs = len(json.loads(r[1] or "[]"))
                    except (json.JSONDecodeError, TypeError):
                        n_msgs = 0
                    try:
                        updated_iso = datetime.fromtimestamp(float(r[2]), tz=timezone.utc).isoformat()
                    except (TypeError, ValueError, OSError):
                        updated_iso = None
                    recent.append({"session_id": r[0], "message_count": n_msgs, "updated_at": updated_iso})

            avg_msgs = total_msgs / max(total_sessions, 1)
            return ChatReport(
                total_sessions=int(total_sessions),
                total_messages=int(total_msgs),
                avg_messages_per_session=round(avg_msgs, 2),
                database_path=str(db_file),
                sessions_found=True,
                recent_sessions=recent,
            )
    except Exception:
        return ChatReport(
            total_sessions=0,
            total_messages=0,
            avg_messages_per_session=0.0,
            database_path=str(db_file),
            sessions_found=False,
            recent_sessions=[],
        )



def collect_rag_report(
    *,
    database_path: str | Path = "data/rag/index.sqlite",
    sample_queries: Sequence[str] | None = None,
) -> RagReport:
    """Benchmark RAG SQLite index chunk counts, search latency, and retrieval quality."""
    db_file = Path(database_path)
    if not db_file.is_file():
        return RagReport(
            database_path=str(db_file),
            index_exists=False,
            total_chunks=0,
            sample_queries_tested=0,
            avg_query_latency_ms=0.0,
            sample_results=[],
        )

    queries = list(sample_queries or [
        "transformer attention mechanism",
        "artificial intelligence neural network",
        "python programming language",
    ])

    try:
        rag = SQLiteRagIndex(db_file)
        total_chunks = rag.count

        latencies: list[float] = []
        sample_results: list[dict[str, Any]] = []

        for q in queries:
            start = time.perf_counter()
            results = rag.search(q, top_k=2)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            sample_results.append({
                "query": q,
                "latency_ms": round(elapsed_ms, 2),
                "hits": [
                    {"title": r.title, "score": round(r.score, 3), "snippet": r.description[:80]}
                    for r in results
                ],
            })

        avg_lat = statistics.mean(latencies) if latencies else 0.0
        return RagReport(
            database_path=str(db_file),
            index_exists=True,
            total_chunks=total_chunks,
            sample_queries_tested=len(queries),
            avg_query_latency_ms=round(avg_lat, 2),
            sample_results=sample_results,
        )
    except Exception as error:
        return RagReport(
            database_path=str(db_file),
            index_exists=True,
            total_chunks=0,
            sample_queries_tested=0,
            avg_query_latency_ms=0.0,
            sample_results=[{"error": str(error)}],
        )


def build_system_inference_report(
    *,
    include_generation: bool = True,
    include_serving: bool = True,
    include_chat: bool = True,
    include_rag: bool = True,
    model_config_path: str | Path = "configs/model.gpu.yaml",
    inference_config_path: str | Path = "configs/inference.yaml",
    tokenizer_path: str | Path = "data/tokenizer-finetuning",
    checkpoint_path: str | Path = "checkpoints/finetuning/best.pt",
    sessions_db_path: str | Path = "data/cache/sessions.sqlite",
    rag_db_path: str | Path = "data/rag/index.sqlite",
    device: str = "cpu",
    endpoint: str = "http://localhost:8000",
) -> SystemInferenceReport:
    """Construct a unified report aggregating generation, serving, chat, and RAG."""
    now_iso = datetime.now(timezone.utc).isoformat()
    gen_rep = None
    serv_rep = None
    chat_rep = None
    rag_rep = None

    if include_generation and Path(checkpoint_path).is_file():
        try:
            gen_rep = collect_generation_report(
                model_config_path=model_config_path,
                inference_config_path=inference_config_path,
                tokenizer_path=tokenizer_path,
                checkpoint_path=checkpoint_path,
                device=device,
            )
        except Exception as err:
            gen_rep = GenerationReport(
                model_name="error",
                device=device,
                samples_evaluated=0,
                total_tokens_generated=0,
                avg_tokens_per_second=0.0,
                avg_ttft_seconds=0.0,
                p95_ttft_seconds=0.0,
                peak_vram_mb=None,
                samples=[{"error": str(err)}],
            )

    if include_serving:
        serv_rep = collect_serving_report(
            inference_config_path=inference_config_path,
            endpoint=endpoint,
        )

    if include_chat:
        chat_rep = collect_chat_report(database_path=sessions_db_path)

    if include_rag:
        rag_rep = collect_rag_report(database_path=rag_db_path)

    return SystemInferenceReport(
        timestamp=now_iso,
        generation=gen_rep,
        serving=serv_rep,
        chat=chat_rep,
        rag=rag_rep,
    )


def record_generation_sample(
    *,
    prompt: str,
    text: str,
    prompt_tokens: int,
    tokens: int,
    duration_s: float,
    ttft_s: float,
    tokens_per_second: float,
    finish_reason: str = "stop",
    model_name: str = "Gopi",
    device: str = "cpu",
    report_json_path: str | Path = "reports/system_report.json",
    auto_export_html: bool = True,
) -> None:
    """Incrementally record a generation event into report files."""
    json_path = Path(report_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    sample_dict = {
        "prompt": prompt,
        "generated_text": text.strip(),
        "prompt_tokens": prompt_tokens,
        "tokens": tokens,
        "duration_s": round(duration_s, 4),
        "ttft_s": round(ttft_s, 4),
        "tokens_per_second": round(tokens_per_second, 2),
        "finish_reason": finish_reason,
    }

    report_obj: dict[str, Any] = {}
    if json_path.is_file():
        try:
            report_obj = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            report_obj = {}

    if not report_obj or "generation" not in report_obj:
        report = build_system_inference_report(include_generation=False)
        report_obj = report.to_dict()
        report_obj["generation"] = {
            "model_name": model_name,
            "device": device,
            "samples_evaluated": 0,
            "total_tokens_generated": 0,
            "avg_tokens_per_second": 0.0,
            "avg_ttft_seconds": 0.0,
            "p95_ttft_seconds": 0.0,
            "peak_vram_mb": None,
            "samples": [],
        }

    gen = report_obj.get("generation") or {}
    samples = gen.get("samples") or []
    samples.append(sample_dict)
    # Keep last 50 samples
    samples = samples[-50:]
    gen["samples"] = samples

    tps_list = [s["tokens_per_second"] for s in samples if s.get("tokens_per_second")]
    ttft_list = [s["ttft_s"] for s in samples if s.get("ttft_s")]
    total_tokens = sum(s.get("tokens", 0) for s in samples)

    gen["samples_evaluated"] = len(samples)
    gen["total_tokens_generated"] = total_tokens
    gen["avg_tokens_per_second"] = round(statistics.mean(tps_list), 2) if tps_list else 0.0
    gen["avg_ttft_seconds"] = round(statistics.mean(ttft_list), 4) if ttft_list else 0.0
    if ttft_list:
        sorted_ttft = sorted(ttft_list)
        p95_idx = round((len(sorted_ttft) - 1) * 0.95)
        gen["p95_ttft_seconds"] = round(sorted_ttft[p95_idx], 4)

    report_obj["timestamp"] = datetime.now(timezone.utc).isoformat()
    report_obj["generation"] = gen

    # Update chat and serving snapshots as well
    chat_rep = collect_chat_report()
    report_obj["chat"] = asdict(chat_rep)
    serv_rep = collect_serving_report()
    report_obj["serving"] = asdict(serv_rep)

    json_str = json.dumps(report_obj, indent=2, ensure_ascii=False)
    json_path.write_text(json_str, encoding="utf-8")

    if auto_export_html:
        try:
            from serving.report_router import DASHBOARD_HTML
            html_content = DASHBOARD_HTML.replace(
                "let currentReportData = null;",
                f"let currentReportData = {json_str};\n    window.addEventListener('DOMContentLoaded', () => {{ renderReport(currentReportData); }});"
            )
            json_path.with_suffix(".html").write_text(html_content, encoding="utf-8")
        except Exception:
            pass

