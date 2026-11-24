# `@stateset/agents`

Typed, dependency-free Node.js / TypeScript client for the StateSet Agents API,
**3D Weight Studio & Live RL Cockpit**, and **Streamable HTTP MCP** control plane.

```bash
npm install @stateset/agents
```

## Inference (Messages & OpenAI-Compatible Chat)

```js
import { StateSet } from "@stateset/agents";

const stateset = new StateSet({
  baseURL: "https://agents.example.com",
  apiKey: process.env.STATESET_API_KEY,
});

// Anthropic-style Messages API
const response = await stateset.messages.create({
  model: "Qwen/Qwen3.8-27B",
  messages: [{ role: "user", content: "Where is order 77701?" }],
  max_tokens: 200,
});
console.log(response.content);

// OpenAI-compatible Chat Completions
const completion = await stateset.chat.completions.create({
  model: "Qwen/Qwen3.8-27B",
  messages: [{ role: "user", content: "Hello" }],
});

// Server-Sent Events streaming
for await (const event of stateset.chat.completions.stream({
  model: "Qwen/Qwen3.8-27B",
  messages: [{ role: "user", content: "Hello" }],
})) {
  process.stdout.write(event.choices?.[0]?.delta?.content ?? "");
}
```

## 3D Weight Studio, Live RL Cockpit & 6-Domain Verifiable Sandboxes

Connect to a running `stateset-agents studio --port 8765` instance via `stateset.studio`:

```js
const studio = new StateSet({ baseURL: "http://127.0.0.1:8765" });

// Plan a reproducible Open-Weight RL run (local, Modal, RunPod, K8s, or River AI)
const plan = await studio.studio.planRlRun({
  model: "Qwen/Qwen3.6-35B-A3B-FP8",
  algorithm: "cispo",
  cloud: "river",
});

// Run concurrent verifiable enterprise RL sandboxes
const pool = await studio.studio.runSandboxPool({
  domains: ["commerce", "coding", "sql", "rest_api", "docker_code", "web_agent"],
});

// Inspect RL telemetry, query SQLite WAL ledger, and send live Pause/Rollback controls
const telemetry = await studio.studio.getRlTelemetry({ limit: 40 });
const trajectories = await studio.studio.queryRlTrajectories({ limit: 20 });
await studio.studio.controlRlRun({ action: "pause", reason: "Inspect entropy dip" });

// Autonomous Auto-Heal, Red-Team Immunization, Canary Gate & OTLP Trace Calibration
const traces = await studio.studio.queryTraces({ failed_only: true, limit: 25 });
const calibration = await studio.studio.auditRewardCalibration();
```

## Streamable HTTP MCP Client

Invoke any of StateSet's annotated MCP tools and live `stateset://studio/*` resources over HTTP JSON-RPC via `stateset.mcp`:

```js
await studio.mcp.initialize();
const { tools } = await studio.mcp.listTools();
const result = await studio.mcp.callTool("studio_run_sandbox_pool", {
  domains: ["commerce", "coding", "sql", "rest_api", "docker_code", "web_agent"],
});
const snapshot = await studio.mcp.readResource("stateset://studio/models");
```

Requires Node.js 18 or newer. Pass a custom `fetch` implementation to the
constructor for compatible runtimes, testing, proxies, or instrumentation.
