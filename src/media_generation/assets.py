"""Content-addressed local asset storage for production media APIs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

_ALLOWED_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _signature_matches(data: bytes, mime: str) -> bool:
    head = data[:32]
    if mime == "image/png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if mime == "image/jpeg":
        return head.startswith(b"\xff\xd8\xff")
    if mime == "image/webp":
        return len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    if mime in {"audio/wav", "audio/x-wav"}:
        return len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if mime == "audio/flac":
        return head.startswith(b"fLaC")
    if mime == "audio/mpeg":
        return head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)
    if mime == "video/mp4":
        return len(head) >= 12 and head[4:8] == b"ftyp"
    if mime == "video/webm":
        return head.startswith(b"\x1a\x45\xdf\xa3")
    return False


@dataclass(frozen=True)
class AssetRecord:
    id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    created_at: float
    path: str


class AssetStore:
    def __init__(self, root: str | Path, *, max_bytes: int | None = None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes or os.getenv("GOPI_MEDIA_MAX_ASSET_BYTES", 100 * 1024 * 1024))
        if self.max_bytes < 1:
            raise ValueError("max asset bytes must be positive")

    def put(self, data: bytes, *, mime_type: str, filename: str | None = None) -> AssetRecord:
        if not data:
            raise ValueError("asset body is empty")
        if len(data) > self.max_bytes:
            raise ValueError(f"asset exceeds maximum size of {self.max_bytes} bytes")
        mime = (mime_type or "application/octet-stream").split(";", 1)[0].strip().lower()
        suffix = _ALLOWED_MIME.get(mime)
        if suffix is None:
            raise ValueError(f"unsupported media type: {mime}")
        if not _signature_matches(data, mime):
            raise ValueError(f"asset content does not match declared media type: {mime}")
        digest = hashlib.sha256(data).hexdigest()
        asset_id = f"asset_{digest[:32]}"
        original = Path(filename or f"upload{suffix}").name
        safe = _SAFE_NAME.sub("_", original).strip("._") or f"upload{suffix}"
        if not safe.lower().endswith(suffix):
            safe += suffix
        payload_path = self.root / f"{asset_id}{suffix}"
        meta_path = self.root / f"{asset_id}.json"
        if not payload_path.exists():
            tmp = payload_path.with_suffix(payload_path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(payload_path)
        record = AssetRecord(
            id=asset_id,
            filename=safe,
            mime_type=mime,
            size_bytes=len(data),
            sha256=digest,
            created_at=time.time(),
            path=str(payload_path),
        )
        meta_path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8")
        return record

    def get(self, asset_id: str) -> AssetRecord:
        if not re.fullmatch(r"asset_[0-9a-f]{32}", asset_id):
            raise KeyError(asset_id)
        meta_path = self.root / f"{asset_id}.json"
        if not meta_path.exists():
            raise KeyError(asset_id)
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        record = AssetRecord(**payload)
        path = Path(record.path).resolve()
        if self.root not in path.parents or not path.exists():
            raise KeyError(asset_id)
        return record

    def resolve(self, asset_id: str) -> Path:
        return Path(self.get(asset_id).path)

    def delete(self, asset_id: str) -> bool:
        try:
            record = self.get(asset_id)
        except KeyError:
            return False
        Path(record.path).unlink(missing_ok=True)
        (self.root / f"{asset_id}.json").unlink(missing_ok=True)
        return True

    def cleanup(self, *, older_than_seconds: float) -> int:
        cutoff = time.time() - max(0.0, older_than_seconds)
        removed = 0
        for meta in self.root.glob("asset_*.json"):
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
                if float(payload.get("created_at", 0)) < cutoff:
                    if self.delete(str(payload["id"])):
                        removed += 1
            except Exception:
                continue
        return removed
