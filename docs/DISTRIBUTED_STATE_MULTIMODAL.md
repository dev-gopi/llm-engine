# Distributed state and multimodal runtime

## Shared Redis state

Set `GOPI_DISTRIBUTED_STATE_REDIS_URL` to make horizontally scaled API replicas share rate-limit, idempotency, session and OpenAI-platform lifecycle metadata. `GOPI_DISTRIBUTED_STATE_PREFIX` can isolate deployments. Session state may use a separate `GOPI_SESSION_REDIS_URL` / `GOPI_SESSION_REDIS_PREFIX`.

The semantic cache now attempts a Redis Search HNSW vector index automatically. On plain Redis, it preserves compatibility by falling back to bounded cosine scanning. Per-tenant semantic-cache entry quotas use an atomic Redis sorted-set reservation so independent replicas cannot oversubscribe the configured entry count.

When `RedisPlatformStore` is enabled, metadata is shared through Redis. Uploaded file bytes use `platform_files_dir`; multi-host deployments must point that directory at shared durable storage (for example a shared filesystem/object-store mount).

## Native latent image generation

`configs/diffusion/latent.production.yaml` is now an executable profile rather than an architecture-only planning file. Train the VAE with `scripts/train_vae.py`, then train the text-conditioned latent U-Net and text encoder with `scripts/train_latent_diffusion.py`. The production image endpoint can use a local checkpoint-backed provider when these variables are configured:

- `GOPI_LATENT_IMAGE_CONFIG`
- `GOPI_LATENT_IMAGE_CHECKPOINT`
- `GOPI_LATENT_VAE_CHECKPOINT`
- `GOPI_LATENT_TEXT_ENCODER_CHECKPOINT`
- `GOPI_LATENT_IMAGE_TOKENIZER`

`GOPI_IMAGE_GENERATION_URL` still takes precedence when an external image provider is desired.

## Audio understanding

Configure `GOPI_AUDIO_UNDERSTANDING_MODEL` and optionally `GOPI_AUDIO_UNDERSTANDING_TASK` / `GOPI_AUDIO_UNDERSTANDING_DEVICE`. `/v1/audio/understand` uses the configured Hugging Face audio runtime and can attach an ASR transcript when ASR is configured.

## Voice cloning

Configure `GOPI_VOICE_CLONING_MODEL` for the optional Coqui XTTS runtime. `/v1/audio/voice-clone` requires a consent token and bounded reference duration before invoking the model.

## Standalone video understanding

Configure `GOPI_VIDEO_UNDERSTANDING_MODEL`, with optional task/device variables. `/v1/videos/understand` uses the native vision-language runtime when available and otherwise falls back to the standalone configured video model.

## Audio-only Responses

`/v1/responses` now accepts `modalities: ["audio"]`. The model may use text internally as an intermediate representation, but text deltas/content are suppressed when text was not requested. Audio can also be requested alongside text.

## Realtime and WebRTC

`/v1/realtime` implements a stateful WebSocket event session supporting text input, base64-WAV audio buffering/transcription, response generation and optional TTS audio events. `/v1/realtime/webrtc` offers the same event contract over an optional `aiortc` WebRTC data channel. Install the `realtime` extra to enable WebRTC signaling.

Heavy model/provider dependencies remain lazy and opt-in so existing text-only serving is unchanged.
