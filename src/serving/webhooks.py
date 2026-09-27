"""Signed webhook delivery with bounded retries and a local dead-letter queue."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class WebhookConfig:
    url: str
    secret: str
    max_attempts: int = 4
    timeout_seconds: float = 10.0
    dead_letter_path: str = "data/cache/webhook-dead-letter.jsonl"


class WebhookDelivery:
    def __init__(self, config: WebhookConfig) -> None:
        if not config.url.startswith(("https://", "http://")):
            raise ValueError("webhook URL must be http(s)")
        if not config.secret:
            raise ValueError("webhook secret is required")
        self.config = config

    def _headers(self, body: bytes, timestamp: int) -> dict[str, str]:
        signed = f"{timestamp}.".encode() + body
        signature = hmac.new(self.config.secret.encode(), signed, hashlib.sha256).hexdigest()
        return {"Content-Type": "application/json", "X-Gopi-Timestamp": str(timestamp), "X-Gopi-Signature": f"v1={signature}"}

    async def deliver(self, event: dict) -> bool:
        body = json.dumps(event, separators=(",", ":"), sort_keys=True).encode()
        for attempt in range(1, self.config.max_attempts + 1):
            ts = int(time.time())
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                    response = await client.post(self.config.url, content=body, headers=self._headers(body, ts))
                if 200 <= response.status_code < 300:
                    return True
            except httpx.HTTPError:
                pass
            if attempt < self.config.max_attempts:
                import asyncio
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        path = Path(self.config.dead_letter_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, "failed_at": int(time.time())}) + "\n")
        return False


__all__ = ["WebhookConfig", "WebhookDelivery"]
