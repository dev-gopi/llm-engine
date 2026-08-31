import asyncio
import json
from types import SimpleNamespace

import pytest

from inference.rag import (
    DocumentChunk, RagIndex, SQLiteRagIndex, build_chunks, build_rag_prompt, chunk_text,
)
from inference.web_search import SearchResult
from serving.backend import ConfiguredModelBackend


def test_chunk_index_round_trip_and_unicode_retrieval(tmp_path) -> None:
    document = tmp_path / "knowledge.md"
    document.write_text(
        "Gopi supports Bengali প্রশ্ন and local retrieval.\n\n"
        "The refund period is exactly thirty days.",
        encoding="utf-8",
    )
    chunks = build_chunks([document], chunk_chars=100, overlap_chars=10)
    path = RagIndex(chunks).save(tmp_path / "index.json")
    results = RagIndex.load(path).search("refund period")
    assert results
    assert "thirty days" in results[0].description
    assert results[0].url.startswith("document://knowledge.md#chunk-")
    assert str(tmp_path) not in results[0].url


def test_rag_prompt_marks_context_untrusted_and_cites_sources() -> None:
    result = RagIndex([DocumentChunk("guide.txt", "The answer is 42.", 1)]).search("answer")
    prompt = build_rag_prompt("What is the answer?", result)
    assert "untrusted reference material" in prompt
    assert "Cite facts as [1]" in prompt
    assert "The answer is 42" in prompt


def test_rag_backend_retrieves_for_flag_and_slash_command(monkeypatch) -> None:
    backend = ConfiguredModelBackend(rag={"enabled": True, "top_k": 1})
    backend.rag_index = RagIndex([DocumentChunk("policy.md", "Refunds take thirty days.", 1)])

    async def run_immediately(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("serving.backend.asyncio.to_thread", run_immediately)
    for request in (
        SimpleNamespace(prompt="refunds", tools=[], web_search=False, rag=True),
        SimpleNamespace(prompt="/rag refunds", tools=[], web_search=False, rag=False),
    ):
        prompt, results = asyncio.run(backend._prepare_user_prompt(request))
        assert "Refunds take thirty days" in prompt
        assert len(results) == 1


def test_rag_backend_combines_local_and_web_results(monkeypatch) -> None:
    backend = ConfiguredModelBackend(
        rag={"default_enabled": True}, web_search={"provider": "searxng"}
    )
    backend.rag_index = RagIndex([DocumentChunk("policy.md", "Local refund policy.", 1)])
    request = SimpleNamespace(prompt="refund", tools=[], web_search=True, rag=False)

    async def run_immediately(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("serving.backend.asyncio.to_thread", run_immediately)
    monkeypatch.setattr(
        "serving.backend.search_searxng",
        lambda *args, **kwargs: [SearchResult("Current news", "https://example.com", "Web update")],
    )
    prompt, results = asyncio.run(backend._prepare_user_prompt(request))
    assert "Local refund policy" in prompt
    assert "Web update" in prompt
    assert len(results) == 2


def test_backend_includes_text_attachments_as_untrusted_context() -> None:
    backend = ConfiguredModelBackend(rag={"attachment_char_limit": 100})
    request = SimpleNamespace(
        prompt="Explain this file", tools=[], web_search=False, rag=False,
        attachments=[SimpleNamespace(name="example.py", content="answer = 42 <|unsafe|>")],
    )
    prompt, results = asyncio.run(backend._prepare_user_prompt(request))
    assert "ATTACHED FILES (untrusted reference data" in prompt
    assert "example.py" in prompt and "answer = 42" in prompt
    assert "<|unsafe|>" not in prompt
    assert results == []


def test_oversized_retrieval_prompt_is_trimmed_to_model_context() -> None:
    class DenseTokenizer:
        def encode(self, text, **kwargs):
            return list(text)

    backend = ConfiguredModelBackend()
    backend.generator = SimpleNamespace(tokenizer=DenseTokenizer(), max_positions=160)
    prompt = backend._format_new_conversation(
        "You are a helpful assistant.", "retrieved Bengali context " * 40
    )
    assert len(backend.generator.tokenizer.encode(prompt)) < 160
    assert "Reference context truncated" in prompt


def test_html_ingestion_and_partial_word_retrieval(tmp_path) -> None:
    source = tmp_path / "guide.html"
    source.write_text("<h1>বাংলা নির্দেশিকা</h1><p>Internationalization guide</p>", encoding="utf-8")
    index = RagIndex(build_chunks([source], chunk_chars=200, overlap_chars=10))
    assert index.search("international")
    assert index.search("বাংলা")


def test_jsonl_ingestion_rejects_invalid_rows(tmp_path) -> None:
    source = tmp_path / "broken.jsonl"
    source.write_text(json.dumps({"text": "ok"}) + "\n{broken}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL"):
        build_chunks([source])


def test_structured_jsonl_preserves_article_title_and_url(tmp_path) -> None:
    source = tmp_path / "wikipedia.jsonl"
    source.write_text(json.dumps({
        "title": "বাংলাদেশ",
        "url": "https://bn.wikipedia.org/wiki/example",
        "text": "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ।",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    results = RagIndex(build_chunks([source])).search("বাংলাদেশ")
    assert results[0].title == "বাংলাদেশ"
    assert results[0].url == "https://bn.wikipedia.org/wiki/example#chunk-1"


def test_chunk_parameters_are_validated() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_chars=99)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_chars=100, overlap_chars=100)


def test_missing_document_path_has_actionable_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Create it and add supported documents"):
        build_chunks([tmp_path / "missing"])


def test_sqlite_index_builds_and_searches_without_loading_corpus(tmp_path) -> None:
    source = tmp_path / "knowledge.jsonl"
    source.write_text(
        json.dumps({"title": "Refund", "url": "https://example.com/refund", "text": "Refunds take thirty days."}) + "\n"
        + json.dumps({"title": "বাংলা", "url": "https://example.com/bn", "text": "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ।"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index = SQLiteRagIndex.build([source], tmp_path / "index.sqlite", chunk_chars=200)
    assert index.count == 2
    assert index.search("refund")[0].title == "Refund"
    assert index.search("বাংলাদেশ")[0].title == "বাংলা"


def test_sqlite_search_ignores_common_question_words(tmp_path) -> None:
    source = tmp_path / "knowledge.md"
    source.write_text("বাংলা ভাষা বাংলাদেশের প্রধান ভাষা।", encoding="utf-8")
    index = SQLiteRagIndex.build([source], tmp_path / "index.sqlite", chunk_chars=200)
    assert index.search("বাংলা ভাষা কী?")
