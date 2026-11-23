# StateSet Open-Weight RL Architecture Guide (`v0.90`–`v0.95`)

StateSet Agents combines **memory-efficient long-context RL** (with optional Triton fused CUDA SRAM kernel), **disaggregated async rollout throughput** (with vLLM Level-1 sleep/wake weight sync), **concurrent multi-domain verifiable MCP tool sandboxes**, **real PyTorch Step-0 surgical warm-start**, **online reward-hacking defense**, and a **Live 3D Weight Studio RL Cockpit** (with MCP & TypeScript SDK parity) in one unified open-weight stack.

---

## 1. Memory-Efficient Chunked GSPO / DAPO / GEPO / VAPO Loss + Triton CUDA Kernel

Large-vocabulary open-weight models (e.g., Qwen3 `V=152,064`, 16k–32k context) normally allocate **37+ GB** just to materialize `[B, T, V]` logits during RL loss computation.

Enable chunked LM-head projection directly across all 4 flagship trainers (`GSPOConfig`, `DAPOConfig`, `GEPOConfig`, `VAPOConfig`):

```python
from stateset_agents.training import (
    DAPOConfig,
    GEPOConfig,
    GSPOConfig,
    VAPOConfig,
    triton_chunked_logprobs_available,
)

print("Triton CUDA kernel available:", triton_chunked_logprobs_available())

gspo_cfg = GSPOConfig(
    model_name="Qwen/Qwen3-8B",
    chunked_loss_chunk_size=256,
    use_loss_checkpointing=True,
)
dapo_cfg = DAPOConfig(
    model_name="Qwen/Qwen3-8B",
    chunked_loss_chunk_size=256,
    use_loss_checkpointing=True,
)
```

### Verified Shootout Benchmarks (`benchmarks/chunked_rl_shootout.py`)

| Benchmark | Measured Result |
| --- | --- |
| Chunked vs. Full-Logit Forward/Backward Parity | `exact_match = True` (`max Δlp = 0.0`, `max Δh_grad = 0.0`, `max Δw_grad = 1.19e-6`) |
| 16k Qwen3 (`V=152,064`, `B=4`, `chunk=256`) Logit VRAM | **37.123 GB → 0.145 GB (99.61% saved)** |
| Disaggregated Async RL + DAPO Dynamic Sampling | **7,729.7 rollouts/s** (`371k tok/s`, zero-variance groups dropped automatically) |
| Concurrent `MultiDomainSandboxPool` (`commerce` + `coding` + `sql`) | **122.7 episodes/s** (`48.89 ms` for 6 multi-turn verifiable episodes w/ per-domain z-score normalization) |

Reproduce locally anytime:

```bash
python benchmarks/chunked_rl_shootout.py --json
# or via CLI:
stateset-agents rl --run-shootout --json
```

---

## 2. Step-0 PyTorch Forward-Hook CAA Surgical Warm-Start

Before starting RL steps, mine a contrastive steering vector directly on your live PyTorch causal LM (HuggingFace, PEFT LoRA, or custom transformer) and bake it into the peak-separation layer to skip cold-start reward-zero plateaus:

```python
from stateset_agents.weight_studio import (
    bake_torch_caa_steering,
    extract_torch_caa_vector,
)

caa = extract_torch_caa_vector(model, positive_input_ids, negative_input_ids)
baked = bake_torch_caa_steering(
    model,
    caa["direction_tensor"],
    layer_index=caa["recommended_layer"],
    alpha=0.5,
)
# Revert cleanly at any time:
# baked["rollback"]()
```

---

## 3. MCP-Native Verifiable Multi-Turn RL Environments

Turn **any** MCP server (in-process callable or live Streamable HTTP MCP URL) into a multi-turn verifiable RL environment with step-level process rewards (PRM) and terminal state verification:

```python
from stateset_agents.core.environment import (
    CodeVerificationSandbox,
    MCPRLEnvironment,
    create_environment,
)

# Built-in deterministic SQLite Commerce Ledger Sandbox
commerce_env = create_environment("mcp", {"max_turns": 6})

# Built-in timeout-guarded Python Coding Verification Sandbox
code_box = CodeVerificationSandbox(timeout_seconds=5.0)
coding_env = MCPRLEnvironment(
    tool_executor=code_box.call_tool,
    state_verifier=code_box.verify,
    max_turns=6,
)

# Or point directly at any live Streamable HTTP MCP server
remote_env = MCPRLEnvironment(
    mcp_http_url="http://127.0.0.1:8765/api/weight-studio/mcp",
    mcp_bearer_token="...",
)
```

---

## 4. Turn-Level Process Reward Credit Assignment

Credit the exact tool-calling turn that succeeded or failed instead of smearing one terminal scalar across an entire trajectory:

```python
from stateset_agents.training.turn_credit import (
    compute_turn_level_advantages,
    expand_turn_advantages_to_tokens,
)

credits = compute_turn_level_advantages(
    [
        {"turn_process_rewards": [1.0, 1.0, 0.9], "terminal_reward": 1.0},
        {"turn_process_rewards": [1.0, 0.0, 0.2], "terminal_reward": 0.0},
    ],
    gamma=0.98,
    prm_weight=0.35,
)
token_adv = expand_turn_advantages_to_tokens(
    [c.turn_advantages for c in credits],
    turn_token_spans=[[(4, 10), (12, 18), (20, 26)], [(4, 10), (12, 18), (20, 26)]],
    seq_len=32,
    as_torch=True,
)
```

---

## 5. Disaggregated Async RL, vLLM Sleep/Wake Sync, Multi-Domain Sandbox Pool & Live 3D RL Cockpit

```python
from stateset_agents.core.environment import MultiDomainSandboxPool, SandboxEpisodeSpec
from stateset_agents.training.disaggregated_rl import DisaggregatedRLConfig, DisaggregatedRLRuntime
from stateset_agents.training.studio_callback import StudioTelemetryCallback

# Concurrent verifiable commerce + coding + SQL sandboxes with per-domain z-score advantages
pool = MultiDomainSandboxPool(max_concurrency=8)

# vLLM Level-1 sleep/wake zero-OOM full-weight sync
rt = DisaggregatedRLRuntime(DisaggregatedRLConfig(live_vllm_endpoint="http://127.0.0.1:8000/v1"))
# rt.sync_vllm_sleep_wake(level=1, weight_push_fn=broadcast_weights)

# Stream live RL steps & OnlineRewardHackingGuard verdicts to 3D Weight Studio Cockpit
cb = StudioTelemetryCallback(run_name="qwen3-8b-gspo", studio_url="http://127.0.0.1:8765")
```

### MCP Tools & TypeScript SDK Parity

External agent harnesses and dashboards can plan RL runs, inspect live cockpit telemetry/controls, and run verifiable multi-domain sandbox pools via MCP (`studio_rl_telemetry`, `studio_rl_plan`, `studio_run_sandbox_pool`) or `@stateset/agents`:

```ts
const telemetry = await client.studio.getRlTelemetry({ limit: 40 });
const plan = await client.studio.planRlRun({ model: "Qwen/Qwen3-8B", algorithm: "dapo", cloud: "modal" });
const pool = await client.studio.runSandboxPool({ domains: ["commerce", "coding", "sql"] });
await client.studio.controlRlRun({ action: "pause", reason: "Inspect entropy dip" });
```

### 1-Command CLI Blueprints & End-to-End Demo

```bash
# Build a reproducible cloud/local RL launch plan + mine failing SQLite OTLP traces into a curriculum
stateset-agents rl --model Qwen/Qwen3-8B --algorithm dapo --cloud modal --gpus 4 \
  --curriculum-from-traces --warm-start --json

# Run the complete offline end-to-end demo (PyTorch CAA warm-start + MCP sandboxes + Chunked GSPO)
python examples/open_weight_rl_end_to_end.py
```

See also: [`docs/WEIGHT_STUDIO.md`](WEIGHT_STUDIO.md) and [`docs/MCP_SERVER.md`](MCP_SERVER.md).
