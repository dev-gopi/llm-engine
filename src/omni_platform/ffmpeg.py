"""Safe FFmpeg wrapper using argument arrays, explicit timeouts, and no shell interpolation."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .errors import DependencyMissingError, MediaValidationError


@dataclass(frozen=True, slots=True)
class FFmpegConfig:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    timeout_seconds: float = 120.0


class FFmpeg:
    def __init__(self, config: FFmpegConfig | None = None) -> None:
        self.config = config or FFmpegConfig()
        if shutil.which(self.config.ffmpeg) is None or shutil.which(self.config.ffprobe) is None:
            raise DependencyMissingError("ffmpeg and ffprobe are required for this media operation")

    def _run(self, args: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(v) for v in args],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout or self.config.timeout_seconds,
            shell=False,
        )

    def probe(self, path: str | Path) -> dict:
        p = Path(path)
        if not p.is_file():
            raise MediaValidationError("media file does not exist")
        cp = self._run([
            self.config.ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)
        ])
        return json.loads(cp.stdout or "{}")

    def transcode(self, src: str | Path, dst: str | Path, *, codec: str | None = None) -> Path:
        src_p, dst_p = Path(src), Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        args = [self.config.ffmpeg, "-y", "-i", str(src_p)]
        if codec:
            args += ["-c:v", codec]
        args += [str(dst_p)]
        self._run(args)
        return dst_p

    def extract_thumbnail(self, src: str | Path, dst: str | Path, *, at_seconds: float = 0.0) -> Path:
        dst_p = Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        self._run([
            self.config.ffmpeg, "-y", "-ss", str(max(0.0, at_seconds)), "-i", str(src), "-frames:v", "1", str(dst_p)
        ])
        return dst_p

    def extract_audio(self, src: str | Path, dst: str | Path, *, codec: str = "pcm_s16le") -> Path:
        dst_p = Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        self._run([self.config.ffmpeg, "-y", "-i", str(src), "-vn", "-c:a", codec, str(dst_p)])
        return dst_p
