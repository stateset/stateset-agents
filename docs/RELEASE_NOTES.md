# Release Notes — StateSet Agents (`v0.99.1` RC → `v1.0.0` GA)

> The authoritative per-release record lives in [`CHANGELOG.md`](../CHANGELOG.md).

## Current Release Candidate: `v0.99.1` (Final `v1.0.0` GA RC)

**Date:** 2026-11-23

### Highlights

- **Open-Weight RL Leadership Suite:** Chunked/Triton LM-head policy loss across all four flagship trainers (`GSPO`, `DAPO`, `GEPO`, `VAPO`) saving `99.61%` peak logit VRAM on 16k Qwen3 sequences, MoE router load-balancing & FP8 rollout TIS/IcePop precision bridge, disaggregated async rollouts with vLLM Level-1 sleep/wake sync, Turn-Level PRM credit assignment, Step-0 PyTorch CAA warm-start, Adaptive Difficulty Curriculum, and Online Reward-Hacking Guard.
- **Six Concurrent Verifiable Enterprise MCP Sandboxes:** `MultiDomainSandboxPool` covering `commerce`, `coding`, `sql`, `rest_api`, `docker_code`, and `web_agent` (`BrowserDOMSandbox`), verified `6/6` in `114.64 ms`.
- **Interactive WebGL2 3D Weight Studio & MCP Control Plane:** Zero-download Hugging Face HTTP-Range `.safetensors` inspector, NF4/AWQ/FP8/SmoothQuant quantization lab, SAE feature clamping, Live RL Cockpit with SQLite WAL ledger, multi-run sparklines, Trajectory Credit Inspector, cooperative 4-trainer Pause/Rollback enforcement, 39 annotated MCP tools (stdio + Streamable HTTP), and `@stateset/agents` TypeScript SDK parity.
- **River AI `0.11` Full-Lifecycle Integration:** Dedicated inference deployments (`RiverDeploymentManager`, `RiverDeploymentProvider`, `river-deploy`, `river-deployments`), streaming replicas, direct checkpoint chat (`RiverInferenceClient`, `river-chat`), `dro`/`decoupled_ppo`, MoE routing replay & expert-flip telemetry, `Session.attest_training_data`, `get_server_capabilities`, `RLBlueprintConfig(cloud="river")`, and 6-domain `river_mcp_sandbox_factory` for River's `rl.RolloutEngine`.
- **Supply-Chain & Verification Discipline:** Frozen `contracts/public_api_v1.json`, `0` `pip-audit` CVEs across locked environments (`[sast]` Semgrep isolation), `0` Bandit Medium/High/Critical findings, `0` strict `mypy --all` errors, and 9,150+ tests (`72.95%` coverage).

### Install

```bash
pip install "stateset-agents==0.99.1"
pip install "stateset-agents[api,mcp]==0.99.1"
pip install "stateset-agents[river]==0.99.1"
npm install @stateset/agents@0.99.1
```

### Links

- Full Changelog: [`CHANGELOG.md`](../CHANGELOG.md)
- PyPI: https://pypi.org/project/stateset-agents/
- npm: https://www.npmjs.com/package/@stateset/agents
- Docs: https://docs.stateset.com · https://docs.stateset.ai
- Issues: https://github.com/stateset/stateset-agents/issues
