"""Unified artifact metadata and storage backends."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .errors import ArtifactNotFoundError, DependencyMissingError


@dataclass(slots=True)
class ArtifactRecord:
    id: str
    job_id: str | None
    type: str
    mime_type: str
    size_bytes: int
    checksum_sha256: str
    storage_key: str
    created_at: float
    expires_at: float | None = None
    codec: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sample_rate: int | None = None
    model: str | None = None
    generation_parameters: dict | None = None


class ArtifactStorage(Protocol):
    def put(self, source: Path, record: ArtifactRecord) -> ArtifactRecord: ...
    def resolve(self, artifact_id: str) -> Path: ...
    def delete(self, artifact_id: str) -> None: ...


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class LocalArtifactStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = self.root / "metadata"
        self.data = self.root / "objects"
        self.meta.mkdir(exist_ok=True)
        self.data.mkdir(exist_ok=True)

    def create_record(self, source: str | Path, *, job_id: str | None, type: str, model: str | None = None, generation_parameters: dict | None = None) -> ArtifactRecord:
        src = Path(source)
        artifact_id = f"art_{uuid.uuid4().hex}"
        mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
        key = f"{artifact_id}{src.suffix.lower()}"
        return ArtifactRecord(
            id=artifact_id,
            job_id=job_id,
            type=type,
            mime_type=mime,
            size_bytes=src.stat().st_size,
            checksum_sha256=_sha256(src),
            storage_key=key,
            created_at=time.time(),
            model=model,
            generation_parameters=generation_parameters,
        )

    def put(self, source: Path, record: ArtifactRecord) -> ArtifactRecord:
        if Path(record.storage_key).name != record.storage_key:
            raise ValueError("storage_key must not contain a path")
        dst = self.data / record.storage_key
        shutil.copy2(source, dst)
        (self.meta / f"{record.id}.json").write_text(json.dumps(asdict(record), sort_keys=True), encoding="utf-8")
        return record

    def resolve(self, artifact_id: str) -> Path:
        meta = self.meta / f"{artifact_id}.json"
        if not meta.is_file():
            raise ArtifactNotFoundError()
        record = json.loads(meta.read_text(encoding="utf-8"))
        path = (self.data / record["storage_key"]).resolve()
        if self.data not in path.parents or not path.is_file():
            raise ArtifactNotFoundError()
        return path

    def delete(self, artifact_id: str) -> None:
        meta = self.meta / f"{artifact_id}.json"
        if not meta.exists():
            return
        record = json.loads(meta.read_text(encoding="utf-8"))
        (self.data / record["storage_key"]).unlink(missing_ok=True)
        meta.unlink(missing_ok=True)


class S3ArtifactStorage:
    """S3-compatible artifact backend with persisted metadata and a bounded local cache."""

    def __init__(self, *, bucket: str, prefix: str = "artifacts", endpoint_url: str | None = None, cache_dir: str | Path = ".cache/gopi-artifacts") -> None:
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise DependencyMissingError("install boto3 to use S3 artifact storage") from exc
        self._client = boto3.client("s3", endpoint_url=endpoint_url)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _meta_key(self, artifact_id: str) -> str:
        return f"{self.prefix}/metadata/{artifact_id}.json"

    def put(self, source: Path, record: ArtifactRecord) -> ArtifactRecord:
        object_key = f"{self.prefix}/objects/{record.storage_key}"
        self._client.upload_file(str(source), self.bucket, object_key, ExtraArgs={"ContentType": record.mime_type})
        record.storage_key = object_key
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._meta_key(record.id),
            Body=json.dumps(asdict(record), sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
        return record

    def _load_record(self, artifact_id: str) -> ArtifactRecord:
        try:
            obj = self._client.get_object(Bucket=self.bucket, Key=self._meta_key(artifact_id))
            payload = json.loads(obj["Body"].read().decode("utf-8"))
            return ArtifactRecord(**payload)
        except Exception as exc:
            raise ArtifactNotFoundError() from exc

    def resolve(self, artifact_id: str) -> Path:
        record = self._load_record(artifact_id)
        suffix = Path(record.storage_key).suffix
        local = self.cache_dir / f"{artifact_id}{suffix}"
        if not local.is_file() or local.stat().st_size != record.size_bytes:
            self._client.download_file(self.bucket, record.storage_key, str(local))
        return local

    def delete(self, artifact_id: str) -> None:
        try:
            record = self._load_record(artifact_id)
        except ArtifactNotFoundError:
            return
        self._client.delete_object(Bucket=self.bucket, Key=record.storage_key)
        self._client.delete_object(Bucket=self.bucket, Key=self._meta_key(artifact_id))
        for candidate in self.cache_dir.glob(f"{artifact_id}.*"):
            candidate.unlink(missing_ok=True)
