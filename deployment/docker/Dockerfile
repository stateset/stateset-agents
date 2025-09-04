# ================================
# Multi-stage Dockerfile for StateSet Agents
# ================================

# ================================
# Base stage with common dependencies
# ================================
FROM python:3.11-slim AS base

# Security: Use non-root user
ENV USER=stateset \
    GROUP=stateset \
    UID=1001 \
    GID=1001 \
    HOME=/home/stateset

# Security: Create non-root user and group
RUN groupadd -r -g $GID $GROUP && \
    useradd -r -u $UID -g $GID -d $HOME -s /bin/bash -c "StateSet User" $USER

# Install security updates and essential packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        lsb-release \
        build-essential \
        git \
        libssl-dev \
        libffi-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Security: Install dumb-init for proper signal handling
RUN curl -Lo /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.5/dumb-init_1.2.5_x86_64 && \
    chmod +x /usr/local/bin/dumb-init

# Create application directory with proper permissions
RUN mkdir -p /app && chown -R $USER:$GROUP /app

# Set working directory
WORKDIR /app

# ================================
# Builder stage for dependencies
# ================================
FROM base AS builder

# Copy dependency files for caching
COPY pyproject.toml requirements*.txt ./

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies in virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# Development stage
# ================================
FROM base AS development

# Install development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        vim \
        htop \
        git \
        openssh-client \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder --chown=$USER:$GROUP /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=$USER:$GROUP . .

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Switch to non-root user
USER $USER

# Development ports
EXPOSE 8000 8001 8002

# Health check for development
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development
CMD ["uvicorn", "stateset_agents.api.ultimate_grpo_service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ================================
# Testing stage
# ================================
FROM development AS testing

# Install additional testing dependencies if needed
RUN pip install --no-cache-dir pytest-cov pytest-xdist

# Copy test files
COPY --chown=$USER:$GROUP tests/ ./tests/

# Run tests by default
CMD ["pytest", "--cov=stateset_agents", "--cov-report=html", "--cov-report=xml"]

# ================================
# Production stage
# ================================
FROM base AS production

# Security: Remove unnecessary packages
RUN apt-get update && \
    apt-get remove -y --purge \
        build-essential \
        git \
        curl \
        && apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder --chown=$USER:$GROUP /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code (excluding development files)
COPY --chown=$USER:$GROUP pyproject.toml ./
COPY --chown=$USER:$GROUP stateset_agents/ ./stateset_agents/
COPY --chown=$USER:$GROUP grpo_agent_framework/ ./grpo_agent_framework/

# Install production package
RUN pip install --no-cache-dir -e .

# Security: Create secure temporary directories
RUN mkdir -p /tmp && chown -R $USER:$GROUP /tmp

# Set secure environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TMPDIR=/tmp \
    PYTHONHASHSEED=random

# Switch to non-root user
USER $USER

# Production ports
EXPOSE 8000

# Security: Use dumb-init as entrypoint
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]

# Health check for production
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["uvicorn", "stateset_agents.api.ultimate_grpo_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ================================
# GPU-enabled production stage
# ================================
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 AS gpu-production

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        curl \
        ca-certificates \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user
ENV USER=stateset \
    GROUP=stateset \
    UID=1001 \
    GID=1001 \
    HOME=/home/stateset

RUN groupadd -r -g $GID $GROUP && \
    useradd -r -u $UID -g $GID -d $HOME -s /bin/bash -c "StateSet User" $USER

# Install dumb-init
RUN curl -Lo /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.5/dumb-init_1.2.5_x86_64 && \
    chmod +x /usr/local/bin/dumb-init

# Create application directory
RUN mkdir -p /app && chown -R $USER:$GROUP /app
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=$USER:$GROUP /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=$USER:$GROUP pyproject.toml ./
COPY --chown=$USER:$GROUP stateset_agents/ ./stateset_agents/
COPY --chown=$USER:$GROUP grpo_agent_framework/ ./grpo_agent_framework/

# Install production package
RUN pip install --no-cache-dir -e .

# GPU-specific environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    TORCH_USE_CUDA_DSA=1

# Switch to non-root user
USER $USER

# GPU production ports
EXPOSE 8000

# GPU health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# GPU production command
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["uvicorn", "stateset_agents.api.ultimate_grpo_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
