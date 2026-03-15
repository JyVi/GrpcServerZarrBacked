# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# install dependencies into /app/.venv — no project install yet
RUN uv sync --frozen --no-install-project

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# grpcio needs these at runtime on slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy the venv from builder
COPY --from=builder /app/.venv /app/.venv

# copy source
COPY src/ ./src/
COPY proto/ ./proto/

# zarr data volume mount point
RUN mkdir -p /app/data

# make the venv the active python
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

# gRPC port
EXPOSE 50051

# run the server
CMD ["python", "src/Server/ServerService.py"]