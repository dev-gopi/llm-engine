"""Reusable high-level runtimes for checkpoint-backed audio/video generation."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import torch

from audio_generation.io import load_wav, save_wav
from audio_generation.longform import crossfade_chunks, plan_windows
from audio_generation.model import AudioDiffusionModel
from audio_generation.pipeline import AudioGenerationPipeline
from diffusion.scheduler import DiffusionScheduler
from media_generation.conditioning import build_text_conditioner
from media_generation.manifest import build_manifest, save_manifest
from media_generation.presets import apply_style, resolve_preset
from media_generation.production import configure_torch_runtime, validate_generation_config
from media_generation.quality import audio_diagnostics, video_diagnostics
from media_generation.progress import CancellationToken, ProgressCallback
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from video_generation.io import image_to_video_tensor, load_image, load_video, save_mp4
from video_generation.longform import blend_video_segments
from video_generation.model import VideoDiffusionModel
from video_generation.pipeline import VideoGenerationPipeline


class AudioGenerator:
    def __init__(self, config_path: str | Path, checkpoint: str | Path, *, device: str | torch.device | None = None):
        self.config_path = Path(config_path)
        self.checkpoint = Path(checkpoint)
        self.config = load_yaml(self.config_path)
        validate_generation_config(self.config, kind="audio")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        configure_torch_runtime(self.config, self.device)
        self.tokenizer = Tokenizer.load(Path(self.config["tokenizer"]))
        self.model = AudioDiffusionModel.from_config(self.config, text_conditioner=build_text_conditioner(self.config, self.tokenizer)).to(self.device)
        load_checkpoint(self.checkpoint, self.model, map_location=self.device, use_ema=True, expected_tokenizer_fingerprint=self.tokenizer.fingerprint)
        self.model.eval()
        self.scheduler = DiffusionScheduler(int(self.config.get("timesteps", 1000)), float(self.config.get("beta_start", 1e-4)), float(self.config.get("beta_end", 2e-2)), device=self.device, schedule=str(self.config.get("noise_schedule", "cosine")))
        self.pipeline = AudioGenerationPipeline(self.model, self.scheduler)

    def _condition(self, prompt: str, negative_prompt: str):
        positive = self.model.text_conditioner.encode_prompts([prompt], self.tokenizer, device=self.device, return_features=True)
        negative = self.model.text_conditioner.encode_prompts([negative_prompt], self.tokenizer, device=self.device, return_features=True) if negative_prompt else None
        return positive, negative

    def generate(self, *, prompt: str, output: str | Path, negative_prompt: str = "", seconds: float | None = None,
                 seed: int | None = 42, preset: str | None = None, style: str | None = None,
                 steps: int | None = None, guidance_scale: float | None = None, init_audio: str | Path | None = None,
                 strength: float = 1.0, preserve_start_seconds: float | None = None, preserve_end_seconds: float | None = None,
                 metadata: bool = True, long_form: bool = False, overlap_seconds: float = 0.5,
                 progress_callback: ProgressCallback | None = None,
                 cancellation_token: CancellationToken | None = None) -> tuple[Path, dict[str, Any]]:
        started = time.perf_counter()
        prompt = apply_style(prompt, style)
        chosen = resolve_preset(preset, self.config)
        steps = int(steps if steps is not None else chosen.steps)
        guidance_scale = float(guidance_scale if guidance_scale is not None else chosen.guidance_scale)
        sample_rate = int(self.config["sample_rate"])
        seconds = float(seconds if seconds is not None else self.config.get("generation_seconds", self.config["duration_seconds"]))
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        positive, negative = self._condition(prompt, negative_prompt)
        target_samples = round(sample_rate * seconds)
        window_samples = round(sample_rate * float(self.config.get("duration_seconds", seconds)))
        init = load_wav(init_audio, sample_rate=sample_rate, samples=target_samples, crop="center", peak_normalize=bool(self.config.get("peak_normalize", True))) if init_audio else None
        preserve_mask = None
        if init is not None and (preserve_start_seconds is not None or preserve_end_seconds is not None):
            preserve_mask = torch.zeros(target_samples)
            if preserve_start_seconds:
                preserve_mask[: min(target_samples, round(sample_rate * preserve_start_seconds))] = 1
            if preserve_end_seconds:
                preserve_mask[max(0, target_samples - round(sample_rate * preserve_end_seconds)):] = 1
        common = dict(context=positive.context, context_mask=positive.mask,
                      negative_condition=None if negative is None else negative.pooled,
                      negative_context=None if negative is None else negative.context,
                      negative_context_mask=None if negative is None else negative.mask,
                      guidance_scale=guidance_scale, inference_steps=steps, eta=chosen.eta,
                      cancellation_token=cancellation_token)
        if long_form and target_samples > window_samples and init is None:
            overlap = max(0, round(sample_rate * overlap_seconds))
            chunks=[]
            windows = plan_windows(target_samples, window_samples, overlap)
            total_windows = max(1, len(windows))
            for i, _ in enumerate(windows):
                def _chunk_progress(done: int, total: int, *, _i=i) -> None:
                    if progress_callback is not None:
                        progress_callback(_i * total + done, total_windows * total)
                chunk = self.pipeline.sample(positive.pooled, samples=window_samples, sample_rate=sample_rate, device=self.device, seed=None if seed is None else seed+i, progress_callback=_chunk_progress, **common)[0]
                chunks.append(chunk)
            audio = crossfade_chunks(chunks, overlap)[:target_samples]
        else:
            audio = self.pipeline.sample(positive.pooled, samples=target_samples, sample_rate=sample_rate, device=self.device, seed=seed, init_waveform=init, strength=strength, preserve_mask=preserve_mask, progress_callback=progress_callback, **common)[0]
        path = save_wav(output, audio, sample_rate)
        settings={"seconds": seconds, "sample_rate": sample_rate, "steps": steps, "guidance_scale": guidance_scale, "preset": chosen.name, "style": style, "strength": strength, "long_form": long_form}
        result={"path": str(path), "quality": audio_diagnostics(audio), "settings": settings}
        if metadata:
            manifest=build_manifest(kind="audio", output=path, prompt=prompt, negative_prompt=negative_prompt, seed=seed, checkpoint=self.checkpoint, config_path=self.config_path, settings=settings, elapsed_seconds=time.perf_counter()-started)
            manifest["quality"] = result["quality"]
            result["manifest"] = str(save_manifest(path.with_suffix(path.suffix+".json"), manifest))
        return path, result


class VideoGenerator:
    def __init__(self, config_path: str | Path, checkpoint: str | Path, *, device: str | torch.device | None = None):
        self.config_path = Path(config_path); self.checkpoint = Path(checkpoint)
        self.config = load_yaml(self.config_path); validate_generation_config(self.config, kind="video")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); configure_torch_runtime(self.config, self.device)
        self.tokenizer = Tokenizer.load(Path(self.config["tokenizer"]))
        self.model = VideoDiffusionModel.from_config(self.config, text_conditioner=build_text_conditioner(self.config, self.tokenizer)).to(self.device)
        load_checkpoint(self.checkpoint, self.model, map_location=self.device, use_ema=True, expected_tokenizer_fingerprint=self.tokenizer.fingerprint)
        self.model.eval()
        self.scheduler=DiffusionScheduler(int(self.config.get("timesteps",1000)), float(self.config.get("beta_start",1e-4)), float(self.config.get("beta_end",2e-2)), device=self.device, schedule=str(self.config.get("noise_schedule","cosine")))
        self.pipeline=VideoGenerationPipeline(self.model,self.scheduler)

    def _condition(self,prompt,negative_prompt):
        p=self.model.text_conditioner.encode_prompts([prompt],self.tokenizer,device=self.device,return_features=True)
        n=self.model.text_conditioner.encode_prompts([negative_prompt],self.tokenizer,device=self.device,return_features=True) if negative_prompt else None
        return p,n

    def generate(self, *, prompt: str, output: str | Path, negative_prompt: str = "", frames: int | None = None,
                 height: int | None = None, width: int | None = None, fps: int | None = None, seed: int | None = 42,
                 preset: str | None = None, style: str | None = None, steps: int | None = None, guidance_scale: float | None = None,
                 init_video: str | Path | None = None, init_image: str | Path | None = None, strength: float = 1.0,
                 crf: int = 18, metadata: bool = True, segments: int = 1, overlap_frames: int = 2,
                 scene_prompts: list[str] | None = None,
                 progress_callback: ProgressCallback | None = None,
                 cancellation_token: CancellationToken | None = None) -> tuple[Path, dict[str, Any]]:
        if init_video and init_image:
            raise ValueError("use either init_video or init_image, not both")
        if segments <= 0:
            raise ValueError("segments must be positive")
        started=time.perf_counter(); prompt=apply_style(prompt,style); chosen=resolve_preset(preset,self.config)
        steps=int(steps if steps is not None else chosen.steps); guidance_scale=float(guidance_scale if guidance_scale is not None else chosen.guidance_scale)
        frames=int(frames or self.config["frames"]); height=int(height or self.config["height"]); width=int(width or self.config["width"]); fps=int(fps or self.config.get("fps",12))
        scene_prompts = [item.strip() for item in (scene_prompts or []) if item and item.strip()]
        if scene_prompts and len(scene_prompts) != segments:
            raise ValueError("scene_prompts length must match segments")
        positive,negative=self._condition(prompt,negative_prompt)
        source=None
        if init_video:
            source=load_video(init_video,frames=frames,height=height,width=width,sampling="center_contiguous")
        elif init_image:
            source=image_to_video_tensor(load_image(init_image,height=height,width=width),frames=frames)
        common=dict(context=positive.context,context_mask=positive.mask,negative_condition=None if negative is None else negative.pooled,
                    negative_context=None if negative is None else negative.context,negative_context_mask=None if negative is None else negative.mask,
                    guidance_scale=guidance_scale,inference_steps=steps,eta=chosen.eta,strength=strength,
                    cancellation_token=cancellation_token)
        clips=[]; current_source=source
        for i in range(segments):
            if cancellation_token is not None and cancellation_token.cancelled:
                raise RuntimeError("generation cancelled")
            segment_positive = positive
            if scene_prompts:
                scene_prompt = apply_style(scene_prompts[i], style)
                segment_positive, _ = self._condition(scene_prompt, negative_prompt)
            segment_common = dict(common)
            segment_common["context"] = segment_positive.context
            segment_common["context_mask"] = segment_positive.mask
            def _segment_progress(done: int, total: int, *, _i=i) -> None:
                if progress_callback is not None:
                    progress_callback(_i * total + done, segments * total)
            clip=self.pipeline.sample(segment_positive.pooled,frames=frames,height=height,width=width,device=self.device,seed=None if seed is None else seed+i,init_video=current_source,progress_callback=_segment_progress,**segment_common)[0]
            clips.append(clip)
            if segments>1:
                last=clip[:,-1:].expand(-1,frames,-1,-1).contiguous()
                current_source=last
        video=blend_video_segments(clips,overlap_frames) if len(clips)>1 else clips[0]
        path=save_mp4(output,video,fps=fps,crf=crf)
        settings={"frames_per_segment":frames,"segments":segments,"scene_prompts":scene_prompts or None,"output_frames":int(video.shape[1]),"height":height,"width":width,"fps":fps,"steps":steps,"guidance_scale":guidance_scale,"preset":chosen.name,"style":style,"strength":strength,"crf":crf}
        result={"path":str(path),"quality":video_diagnostics(video),"settings":settings}
        if metadata:
            manifest=build_manifest(kind="video",output=path,prompt=prompt,negative_prompt=negative_prompt,seed=seed,checkpoint=self.checkpoint,config_path=self.config_path,settings=settings,elapsed_seconds=time.perf_counter()-started)
            manifest["quality"]=result["quality"]; result["manifest"]=str(save_manifest(path.with_suffix(path.suffix+".json"),manifest))
        return path,result
