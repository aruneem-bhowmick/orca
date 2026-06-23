# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv==0.6.16

WORKDIR /build

# Copy dependency manifests first for layer caching
COPY packages/orca-shared/pyproject.toml packages/orca-shared/pyproject.toml
COPY packages/orca-web/pyproject.toml packages/orca-web/pyproject.toml

# Resolve and install all production dependencies
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache \
        -r packages/orca-web/pyproject.toml

# Copy source packages and install
COPY packages/orca-shared/ packages/orca-shared/
COPY packages/orca-web/ packages/orca-web/
RUN . /opt/venv/bin/activate && \
    uv pip install --no-cache \
        packages/orca-shared \
        packages/orca-web

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1001 orca && \
    useradd --uid 1001 --gid orca --shell /bin/bash --create-home orca

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=orca:orca packages/orca-web/alembic.ini /app/alembic.ini
COPY --chown=orca:orca packages/orca-web/alembic /app/alembic

USER orca

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8003/health').raise_for_status()"

CMD ["uvicorn", "orca_web.api.main:app", "--host", "0.0.0.0", "--port", "8003"]
