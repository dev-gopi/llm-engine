import asyncio
import io
import json
from types import SimpleNamespace

from inference.web_search import SearchResult, build_search_prompt, search_searxng
from serving.backend import ConfiguredModelBackend


def test_build_search_prompt_numbers_sources_and_marks_them_untrusted() -> None:
    prompt = build_search_prompt(
        "current answer?",
        [SearchResult("Example", "https://example.com", "A useful snippet")],
    )
    assert "untrusted reference material" in prompt
    assert "[1] Example" in prompt
    assert "https://example.com" in prompt


def test_searxng_search_normalizes_results(monkeypatch) -> None:
    payload = {"results": [{"title": "Example", "url": "https://example.com", "content": "Snippet"}]}
    response = io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr("inference.web_search.urlopen", lambda request, timeout: response)
    results = search_searxng("test query", endpoint="http://search.local", max_results=1)
    assert results == [SearchResult("Example", "https://example.com", "Snippet")]


def test_backend_uses_default_path_when_search_finds_nothing(monkeypatch) -> None:
    backend = ConfiguredModelBackend(web_search={"provider": "searxng"})
    request = SimpleNamespace(prompt="missing topic", tools=[], web_search=True)
    monkeypatch.setattr("serving.backend.search_searxng", lambda *args, **kwargs: [])

    async def run_immediately(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("serving.backend.asyncio.to_thread", run_immediately)

    prompt, results = asyncio.run(backend._prepare_user_prompt(request))

    assert prompt is None
    assert results == []
