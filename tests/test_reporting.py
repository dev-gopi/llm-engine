"""Tests for the unified inference/operations reporting module."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inference.reporting import (
    ChatReport,
    GenerationReport,
    RagReport,
    ServingReport,
    SystemInferenceReport,
    build_system_inference_report,
    collect_chat_report,
    collect_rag_report,
    collect_serving_report,
)


def test_standalone_system_report_uses_local_json_and_sample() -> None:
    root = Path(__file__).parents[1]
    html = (root / "reports" / "system_report.html").read_text(encoding="utf-8")
    sample = json.loads((root / "reports" / "system_report.sample.json").read_text(encoding="utf-8"))

    assert "system_report.json" in html
    assert "system_report.sample.json" in html
    assert 'id="generation-modal"' in html
    assert "openGenerationModal" in html
    assert "play-gen-preview" in html
    assert "generation-modal-metrics" in html
    assert sample["generation"]["samples_evaluated"] >= 1
    assert sample["serving"]["paged_kv_pages"] >= 1


# ─── Chat Report ──────────────────────────────────────────────────────────────

class TestChatReport:
    def test_missing_database_returns_empty(self, tmp_path: Path) -> None:
        report = collect_chat_report(database_path=tmp_path / "no.sqlite")
        assert report.total_sessions == 0
        assert report.total_messages == 0
        assert report.sessions_found is False
        assert report.avg_messages_per_session == 0.0

    def test_database_no_sessions_table(self, tmp_path: Path) -> None:
        db = tmp_path / "empty.sqlite"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE other (id INTEGER PRIMARY KEY)")
        report = collect_chat_report(database_path=db)
        assert report.sessions_found is False
        assert report.total_sessions == 0

    def test_database_with_sessions(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.sqlite"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE sessions "
                "(session_id TEXT PRIMARY KEY, message_count INTEGER, updated_at TEXT)"
            )
            conn.executemany(
                "INSERT INTO sessions VALUES (?, ?, ?)",
                [
                    ("s1", 4, "2026-09-25T10:00:00"),
                    ("s2", 8, "2026-09-25T11:00:00"),
                    ("s3", 2, "2026-09-25T12:00:00"),
                ],
            )
        report = collect_chat_report(database_path=db)
        assert report.sessions_found is True
        assert report.total_sessions == 3
        assert report.total_messages == 14
        assert abs(report.avg_messages_per_session - 14 / 3) < 0.01
        assert len(report.recent_sessions) == 3

    def test_report_dataclass_fields(self, tmp_path: Path) -> None:
        report = collect_chat_report(database_path=tmp_path / "x.sqlite")
        assert isinstance(report, ChatReport)
        assert hasattr(report, "database_path")
        assert hasattr(report, "recent_sessions")


# ─── RAG Report ───────────────────────────────────────────────────────────────

class TestRagReport:
    def test_missing_index_returns_not_found(self, tmp_path: Path) -> None:
        report = collect_rag_report(database_path=tmp_path / "no_index.sqlite")
        assert report.index_exists is False
        assert report.total_chunks == 0
        assert report.avg_query_latency_ms == 0.0

    def test_rag_report_with_live_index(self) -> None:
        """Smoke test with actual SQLite RAG index if present."""
        rag_path = Path("data/rag/index.sqlite")
        if not rag_path.is_file():
            pytest.skip("RAG index not present")
        report = collect_rag_report(
            database_path=rag_path,
            sample_queries=["neural network", "python"],
        )
        assert report.index_exists is True
        assert report.total_chunks > 0
        assert report.sample_queries_tested == 2
        assert report.avg_query_latency_ms >= 0.0
        assert len(report.sample_results) == 2

    def test_rag_report_fields(self, tmp_path: Path) -> None:
        report = collect_rag_report(database_path=tmp_path / "no.sqlite")
        assert isinstance(report, RagReport)
        assert hasattr(report, "sample_results")


# ─── Serving Report ───────────────────────────────────────────────────────────

class TestServingReport:
    def test_offline_server_returns_correct_status(self) -> None:
        report = collect_serving_report(
            inference_config_path="configs/inference.yaml",
            endpoint="http://localhost:19999",  # nothing running there
        )
        assert isinstance(report, ServingReport)
        assert report.online is False
        assert report.paged_kv_pages > 0
        assert report.requests_per_minute > 0

    def test_serving_report_memory_estimates_present(self) -> None:
        report = collect_serving_report(
            inference_config_path="configs/inference.yaml",
            endpoint="http://localhost:19999",
        )
        # Memory estimates should be non-negative; model config exists
        assert report.estimated_weight_memory_mb >= 0.0
        assert report.estimated_kv_cache_memory_mb >= 0.0

    def test_serving_report_fields(self) -> None:
        report = collect_serving_report(endpoint="http://localhost:19999")
        assert isinstance(report, ServingReport)
        assert hasattr(report, "model_config_path")
        assert hasattr(report, "metadata")
        assert isinstance(report.metadata, dict)


# ─── Generation Report (mocked) ───────────────────────────────────────────────

class TestGenerationReport:
    def _make_mock_generation_report(self) -> GenerationReport:
        return GenerationReport(
            model_name="Gopi",
            device="cpu",
            samples_evaluated=3,
            total_tokens_generated=42,
            avg_tokens_per_second=14.0,
            avg_ttft_seconds=0.05,
            p95_ttft_seconds=0.08,
            peak_vram_mb=None,
            samples=[
                {"prompt": "Hello", "generated_text": "Hi!", "tokens": 14, "duration_s": 1.0, "ttft_s": 0.05, "tokens_per_second": 14.0},
            ],
        )

    def test_generation_report_fields(self) -> None:
        r = self._make_mock_generation_report()
        assert isinstance(r, GenerationReport)
        assert r.samples_evaluated == 3
        assert r.avg_tokens_per_second > 0.0
        assert isinstance(r.samples, list)

    def test_generation_report_with_checkpoint(self) -> None:
        """Smoke test actual generation if checkpoint exists."""
        ckpt = Path("checkpoints/finetuning/best.pt")
        if not ckpt.is_file():
            pytest.skip("Fine-tuned checkpoint not present")
        from inference.reporting import collect_generation_report
        report = collect_generation_report(
            model_config_path="configs/model.gpu.yaml",
            inference_config_path="configs/inference.yaml",
            tokenizer_path="data/tokenizer-finetuning",
            checkpoint_path=ckpt,
            prompts=["What is AI?"],
            max_tokens=12,
            device="cpu",
        )
        assert isinstance(report, GenerationReport)
        assert report.samples_evaluated == 1
        assert report.total_tokens_generated > 0
        assert report.avg_tokens_per_second > 0.0
        assert len(report.samples) == 1


# ─── SystemInferenceReport ────────────────────────────────────────────────────

class TestSystemInferenceReport:
    def _build_report(self, tmp_path: Path) -> SystemInferenceReport:
        return SystemInferenceReport(
            timestamp="2026-09-25T12:00:00+00:00",
            generation=GenerationReport(
                model_name="Gopi",
                device="cpu",
                samples_evaluated=2,
                total_tokens_generated=30,
                avg_tokens_per_second=10.0,
                avg_ttft_seconds=0.04,
                p95_ttft_seconds=0.06,
                peak_vram_mb=None,
                samples=[],
            ),
            serving=ServingReport(
                status="offline",
                endpoint="http://localhost:8000",
                online=False,
                model_config_path="configs/model.gpu.yaml",
                checkpoint_path="checkpoints/finetuning/best.pt",
                paged_kv_pages=128,
                paged_kv_page_size=16,
                prefix_cache_capacity=16,
                requests_per_minute=60,
                estimated_weight_memory_mb=820.0,
                estimated_kv_cache_memory_mb=64.0,
            ),
            chat=ChatReport(
                total_sessions=5,
                total_messages=40,
                avg_messages_per_session=8.0,
                database_path="data/cache/sessions.sqlite",
                sessions_found=True,
                recent_sessions=[],
            ),
            rag=RagReport(
                database_path="data/rag/index.sqlite",
                index_exists=True,
                total_chunks=313913,
                sample_queries_tested=3,
                avg_query_latency_ms=4.5,
                sample_results=[],
            ),
        )

    def test_to_dict_has_all_keys(self, tmp_path: Path) -> None:
        report = self._build_report(tmp_path)
        d = report.to_dict()
        assert "timestamp" in d
        assert "generation" in d
        assert "serving" in d
        assert "chat" in d
        assert "rag" in d

    def test_to_json_valid_json(self, tmp_path: Path) -> None:
        report = self._build_report(tmp_path)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["generation"]["model_name"] == "Gopi"
        assert parsed["serving"]["paged_kv_pages"] == 128
        assert parsed["chat"]["total_sessions"] == 5
        assert parsed["rag"]["total_chunks"] == 313913

    def test_to_markdown_has_sections(self, tmp_path: Path) -> None:
        report = self._build_report(tmp_path)
        md = report.to_markdown()
        assert "Generation" in md or "generation" in md.lower()
        assert "Serving" in md or "serving" in md.lower()
        assert "Chat" in md or "chat" in md.lower()
        assert "RAG" in md or "rag" in md.lower()

    def test_none_sections_produce_partial_report(self) -> None:
        report = SystemInferenceReport(
            timestamp="2026-09-25T12:00:00+00:00",
            generation=None,
            serving=None,
            chat=None,
            rag=None,
        )
        d = report.to_dict()
        assert d["generation"] is None
        assert d["serving"] is None
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0


# ─── build_system_inference_report ────────────────────────────────────────────

class TestBuildSystemInferenceReport:
    def test_serving_and_chat_and_rag_only(self, tmp_path: Path) -> None:
        """Skip generation; just collect serving/chat/rag quickly."""
        db = tmp_path / "sessions.sqlite"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE sessions "
                "(session_id TEXT, message_count INTEGER, updated_at TEXT)"
            )
        report = build_system_inference_report(
            include_generation=False,
            include_serving=True,
            include_chat=True,
            include_rag=False,
            sessions_db_path=db,
            rag_db_path=tmp_path / "no_rag.sqlite",
            endpoint="http://localhost:19999",
        )
        assert report.generation is None
        assert report.rag is None
        assert isinstance(report.serving, ServingReport)
        assert isinstance(report.chat, ChatReport)

    def test_no_sections_returns_shell_report(self, tmp_path: Path) -> None:
        report = build_system_inference_report(
            include_generation=False,
            include_serving=False,
            include_chat=False,
            include_rag=False,
        )
        assert report.generation is None
        assert report.serving is None
        assert report.chat is None
        assert report.rag is None
        assert isinstance(report.timestamp, str)

    def test_rag_report_included(self) -> None:
        rag_path = Path("data/rag/index.sqlite")
        if not rag_path.is_file():
            pytest.skip("RAG index not available")
        report = build_system_inference_report(
            include_generation=False,
            include_serving=False,
            include_chat=False,
            include_rag=True,
            rag_db_path=rag_path,
        )
        assert isinstance(report.rag, RagReport)
        assert report.rag.index_exists is True
        assert report.rag.total_chunks > 0
