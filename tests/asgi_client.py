"""Synchronous test adapter built on the supported HTTPX ASGI transport."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class ASGIClient:
    """Small context-manager-compatible client for synchronous API tests.

    Unlike Starlette's deprecated ``TestClient`` portal, each request runs in a
    managed ASGI lifespan context using HTTPX's native ASGI transport.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    def __enter__(self) -> ASGIClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async def send() -> httpx.Response:
            async with self.app.router.lifespan_context(self.app):
                transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    return await client.request(method, url, **kwargs)

        return asyncio.run(send())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def options(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("OPTIONS", url, **kwargs)
