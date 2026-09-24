"""Video frame loading and MP4 writing with PyAV/FFmpeg backends."""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def _frame_to_tensor(array: np.ndarray) -> Tensor:
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(127.5).sub(1.0)


def _sample_indices(length: int, frames: int, mode: str) -> list[int]:
    if length <= 0 or frames <= 0:
        raise ValueError("video length and requested frame count must be positive")
    if mode == "uniform":
        return torch.linspace(0, length - 1, frames).round().long().tolist()
    if mode not in {"start_contiguous", "center_contiguous", "random_contiguous"}:
        raise ValueError("sampling must be uniform, start_contiguous, center_contiguous, or random_contiguous")
    if length >= frames:
        available = length - frames
        if mode == "center_contiguous":
            start = available // 2
        elif mode == "random_contiguous":
            start = random.randint(0, available)
        else:
            start = 0
        return list(range(start, start + frames))
    # Repeat the last available frame rather than spreading a short clip apart.
    return list(range(length)) + [length - 1] * (frames - length)


def load_video(
    path: str | Path,
    *,
    frames: int,
    height: int,
    width: int,
    sampling: str = "uniform",
    horizontal_flip: bool = False,
) -> Tensor:
    source = Path(path)
    if min(frames, height, width) <= 0:
        raise ValueError("frames, height, and width must be positive")
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise RuntimeError("video loading requires the optional 'av' package: pip install -e '.[media]'") from exc
    decoded: list[Tensor] = []
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            image = frame.to_image().resize((width, height))
            decoded.append(_frame_to_tensor(np.asarray(image.convert("RGB"))))
    if not decoded:
        raise ValueError(f"no video frames decoded from: {source}")
    indices = _sample_indices(len(decoded), frames, sampling)
    video = torch.stack([decoded[index] for index in indices], dim=1)
    if horizontal_flip:
        video = video.flip(-1)
    return video


def save_mp4(path: str | Path, video: Tensor, *, fps: int = 12, crf: int = 18) -> Path:
    """Save ``[3, frames, H, W]`` video; prefer PyAV and stream raw RGB to FFmpeg fallback."""
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("video must have shape [3, frames, height, width]")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = ((video.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).numpy()
    try:
        import av  # type: ignore
        with av.open(str(destination), mode="w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = int(video.shape[-1])
            stream.height = int(video.shape[-2])
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "preset": "medium", "movflags": "+faststart"}
            for array in arrays:
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return destination
    except Exception:
        destination.unlink(missing_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("MP4 output requires PyAV or ffmpeg")
    height, width = int(video.shape[-2]), int(video.shape[-1])
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    process = subprocess.run(command, input=arrays.tobytes(), check=False)
    if process.returncode != 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed with exit code {process.returncode}")
    return destination


def load_image(path: str | Path, *, height: int, width: int) -> Tensor:
    """Load one RGB image as normalized tensor ``[3,H,W]`` for image-to-video."""
    if min(height, width) <= 0:
        raise ValueError("height and width must be positive")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("image-to-video requires Pillow: pip install pillow") from exc
    image = Image.open(path).convert("RGB").resize((width, height))
    return _frame_to_tensor(np.asarray(image))


def image_to_video_tensor(image: Tensor, *, frames: int) -> Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3,H,W]")
    if frames <= 0:
        raise ValueError("frames must be positive")
    return image[:, None].expand(-1, frames, -1, -1).contiguous()
