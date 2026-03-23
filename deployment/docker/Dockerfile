# syntax=docker/dockerfile:1

# Multi-stage Dockerfile for the StateSet Agents Python API gateway.
#
# This image is intended to run the FastAPI gateway (`stateset_agents.api.main:app`)
# which proxies inference to a backend (e.g. vLLM) and can optionally run training
# workflows. GPU inference and GPU model serving should be done in dedicated
# model-server images (e.g. vLLM) deployed alongside this gateway.
#
# Targets:
# - runtime (default): production-ish gateway
# - dev: dev gateway with extra tooling
# - test: runs the Python test suite

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime deps: curl (health checks) + certs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for runtime.
RUN groupadd -r -g 10001 stateset && \
    useradd -r -u 10001 -g 10001 -m -d /home/stateset stateset


FROM base AS builder

# Build deps for native wheels (kept out of final image).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

ARG STATESET_EXTRAS="api"

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only what we need to install the package. (The `grpo_agent_framework`
# package is intentionally not installed via packaging, but is used at runtime
# through `PYTHONPATH=/app`.)
COPY pyproject.toml setup.py README.md ./
COPY stateset_agents/ ./stateset_agents/
COPY grpo_agent_framework/ ./grpo_agent_framework/

RUN pip install --no-cache-dir ".[${STATESET_EXTRAS}]"


FROM builder AS dev

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

ARG DEV_EXTRAS="api,dev"
RUN pip install --no-cache-dir ".[${DEV_EXTRAS}]"

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "stateset_agents.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


FROM dev AS test

# Test stage needs the full repo for `tests/` and legacy shims.
USER root
COPY . .
USER stateset

CMD ["pytest", "-q"]


FROM base AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

COPY --from=builder /opt/venv /opt/venv

# Runtime-only source needed on PYTHONPATH (not installed as a distribution).
COPY --chown=10001:10001 grpo_agent_framework/ ./grpo_agent_framework/

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "stateset_agents.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
