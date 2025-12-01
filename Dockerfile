# StateSet Agents - Multi-stage Docker Build
# Build stage
FROM rust:1.82-bookworm as builder

WORKDIR /app

# Install protobuf compiler for gRPC
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./

# Create dummy source to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && echo "" > src/lib.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release && rm -rf src target/release/deps/stateset_agents*

# Copy actual source code
COPY src ./src

# Build the application
RUN cargo build --release --features full

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/stateset-agents /app/stateset-agents

# Create non-root user
RUN useradd -r -s /bin/false stateset
USER stateset

# Set environment defaults
ENV RUST_LOG=info
ENV LOG_JSON=true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD pgrep stateset-agents || exit 1

# Run the application
CMD ["/app/stateset-agents"]
