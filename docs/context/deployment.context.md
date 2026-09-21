# Compact Context: Deployment & Containerization (`docs/context/deployment.context.md`)

> **AGENT CONTEXT PACK**: Load this file when working on Docker packaging, Compose services, Nginx reverse proxy, TLS certificates, or model export.

---

## 1. Authoritative Sources of Truth
- **Dockerfile**: [`Dockerfile`](../../Dockerfile)
- **Docker Compose**: [`compose.yaml`](../../compose.yaml), [`compose.gpu.yaml`](../../compose.gpu.yaml)
- **Nginx Config**: [`deploy/nginx.conf`](../../deploy/nginx.conf)
- **Cert Generator**: [`deploy/generate_dev_certs.sh`](../../deploy/generate_dev_certs.sh)
- **Model Exporter**: [`scripts/export.py`](../../scripts/export.py)
- **Full Guide**: [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md)

---

## 2. Containerized Deployment Commands

### GPU Compose Deployment:
```bash
# Generate dev self-signed TLS certs
bash deploy/generate_dev_certs.sh

# Launch GPU-accelerated container
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

### Model Export Commands:
```bash
# Export to Safetensors
.venv/bin/python scripts/export.py --format safetensors --checkpoint checkpoints/finetuning/best.pt

# Export to ONNX
.venv/bin/python scripts/export.py --format onnx --checkpoint checkpoints/finetuning/best.pt
```

---

## 3. Key Invariants
1. Do not bake checkpoint `.pt` files directly into container images; mount them via volumes at `/app/checkpoints`.
2. GPU container requires NVIDIA Container Toolkit installed on host.
3. TLS certificate keys in `deploy/certs/` are gitignored and must never be committed.

---

## 4. Primary Verification Tests
```bash
.venv/bin/pytest tests/test_export.py -q
```

