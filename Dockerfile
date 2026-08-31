# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GOPI_DEVICE=cpu

WORKDIR /app

RUN addgroup --system --gid 10001 gopi \
    && adduser --system --uid 10001 --ingroup gopi --home /app gopi

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY ui ./ui

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && mkdir -p /app/data/cache /app/checkpoints \
    && chown -R gopi:gopi /app/data/cache /app/checkpoints

USER gopi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)" || exit 1

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
