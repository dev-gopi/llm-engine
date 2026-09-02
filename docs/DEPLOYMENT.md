# Deployment guide

The built-in server is suitable for local and trusted-network use. A public
deployment also needs an HTTPS reverse proxy, access controls, resource limits,
monitoring, and operational testing.

## 1. Prepare a final artifact

Use a validated `best.pt`, not `latest.pt`, and follow
[V2_TRAINING_GUIDE.md](V2_TRAINING_GUIDE.md) to evaluate and export it. Keep the
model configuration and tokenizer with the checkpoint.

## 2. Create a dedicated environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --editable .
python scripts/capabilities.py
```

Run the service as a non-root operating-system user. Give it read-only access
to model artifacts and write access only to explicitly configured session,
rate-limit, cache, or log locations.

## 3. Store runtime configuration

Set secrets through the process environment or a secret manager, not source
control:

```bash
export GOPI_API_KEY='replace-with-a-long-random-value'
export GOPI_CHECKPOINT_PATH='checkpoints/dpo/best.pt'
export GOPI_MODEL_CONFIG='configs/model.gpu.yaml'
export GOPI_TOKENIZER_PATH='data/tokenizer-v3'
export GOPI_INFERENCE_CONFIG='configs/inference.yaml'
export GOPI_DEVICE='cuda'
export GOPI_MCP_ENABLED='false'
export GOPI_UID="$(id -u)"
export GOPI_GID="$(id -g)"
export GOPI_MODEL_NAME='gopi'
export GOPI_MAX_CONCURRENCY='1'
export GOPI_REQUESTS_PER_MINUTE='30'
```

Start conservatively on a laptop GPU. Raise concurrency only after load tests
show that VRAM and latency remain safe.

For the container deployment, create the local production environment file:

```bash
cp .env.production.example .env.production
```

Replace `GOPI_API_KEY` with at least 32 random characters. `.env.production` is
ignored by Git; the example file is safe to commit because it contains no real
secret. On Linux, set `GOPI_UID` and `GOPI_GID` to the output of `id -u` and
`id -g`; this lets the non-root API process read host-mounted tokenizer and
checkpoint files without making those model artifacts world-readable.
MCP is disabled by default in Compose because the minimal image does not ship
Node/`npx` or external MCP servers. Build and review those dependencies before
setting `GOPI_MCP_ENABLED=true`.

## 4. Start and verify

```bash
.venv/bin/python scripts/serve.py --host 127.0.0.1 --port 8000
```

From another terminal:

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
```

Keep the application bound to loopback when a same-host reverse proxy handles
public traffic.

## 5. Put a reverse proxy in front

Configure the proxy to:

- terminate TLS with a valid certificate;
- restrict request-body size and connection counts;
- preserve streaming responses without buffering;
- set upstream connect, read, and idle timeouts above the expected generation
  duration;
- protect `/metrics`, `/docs`, `/redoc`, `/openapi.json`, and administrative
  routes;
- forward `Authorization` and `X-Request-ID` securely;
- log metadata without logging prompts, responses, or bearer keys by default.

The application-level API key is a basic control, not a full identity and
authorization system. Use a trusted gateway for multiple users or internet
exposure.

This repository includes `deploy/nginx.conf`. It terminates TLS, supports SSE
and WebSocket streaming, applies connection/request limits and security
headers, and forwards only to the API on an internal Compose network.

## 6. Browser UI and CORS

The bundled UI is available at `/ui/`. For a separate web origin, set
`GOPI_CORS_ORIGINS` to an explicit comma-separated allowlist. Do not use a
wildcard with credentials. Bearer authentication is allowed in CORS preflight
responses for those trusted origins. Server-side third-party
OpenAI-compatible UIs should use base URL `https://YOUR_HOST/v1`, model
`GOPI_MODEL_NAME`, and the bearer key.

## 7. Containers

Mount checkpoints and tokenizers read-only. Do not bake private datasets or API
keys into an image. Expose only the proxy port. If a UI container calls a model
server on the host, configure an explicit host-gateway address; its
`127.0.0.1` refers to the UI container itself.

GPU containers require a compatible host driver, NVIDIA container runtime, and
a PyTorch build compatible with the host. Verify this environment before
starting the service.

Generate a short-lived self-signed certificate for local testing only:

```bash
./deploy/generate_dev_certs.sh
```

Start CPU serving:

```bash
docker compose --env-file .env.production up --build -d
curl --insecure https://localhost:8443/health/ready
```

Start GPU serving with the NVIDIA container runtime installed:

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.gpu.yaml up --build -d
```

The base image installs PyTorch from the official CPU-only wheel index to avoid
downloading CUDA libraries for CPU serving. The GPU Compose override selects
the CUDA-enabled PyPI wheel and therefore produces a substantially larger first
build.

For public deployment, replace `deploy/certs/server.crt` and `server.key` with
files issued by a trusted CA and remove `--insecure` from client commands.
Validate before starting:

```bash
docker compose --env-file .env.production config --quiet
```

## 8. Monitoring and recovery

Monitor readiness, request rate, latency, busy/time-out responses, GPU memory,
process memory, and restarts. Alert on repeated 500, 503, or 504 responses.
Retain a known-good model artifact and test reload/restart procedures before an
upgrade.

Do not overwrite the only checkpoint during deployment. Version artifacts and
deploy a new directory, perform a smoke test, then switch traffic. Roll back to
the previous immutable artifact if readiness or quality checks fail.

## 9. Security and privacy checklist

- use TLS for traffic outside the local machine;
- rotate long random API keys and never put them in URLs;
- restrict administrative and metrics routes at the proxy;
- disable or carefully review MCP, search, and local tools;
- treat prompts, outputs, sessions, and logs as potentially sensitive;
- enforce retention and deletion rules appropriate to the deployment;
- review model and dataset licenses before distribution or commercial use;
- run adversarial and abuse tests for the intended application.

This repository is not a turnkey production platform. Multi-node serving,
tenant isolation, billing, moderation, and compliance controls remain the
operator's responsibility.
