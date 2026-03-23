Architecture
============

The canonical runtime is:

1. Gateway (FastAPI): `stateset_agents.api.main:app`
2. Model server (GPU): vLLM (OpenAI-compatible)

For deployment patterns, see `deployment/helm/stateset-agents` and the Kimi-K2.5
GKE guides in `docs/`.

