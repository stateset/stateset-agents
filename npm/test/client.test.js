import assert from "node:assert/strict";
import test from "node:test";

import { StateSet, StateSetError } from "../src/index.js";

function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body), {
    ...init,
    status: 200,
    headers: { "content-type": "application/json", ...(init.headers || {}) },
    ...(init.status === undefined ? {} : { status: init.status }),
  });
}

test("messages.create sends auth and JSON to the stable endpoint", async () => {
  let captured;
  const client = new StateSet({
    baseURL: "https://agents.example.com/",
    apiKey: "secret",
    fetch: async (url, init) => {
      captured = { url, init };
      return jsonResponse({ id: "msg_1", content: [] });
    },
  });

  const result = await client.messages.create({
    model: "test-model",
    messages: [{ role: "user", content: "hello" }],
  });

  assert.equal(captured.url, "https://agents.example.com/v1/messages");
  assert.equal(captured.init.method, "POST");
  assert.equal(captured.init.headers.get("authorization"), "Bearer secret");
  assert.equal(JSON.parse(captured.init.body).model, "test-model");
  assert.equal(result.id, "msg_1");
});

test("chat.completions.create and models.list use OpenAI-compatible paths", async () => {
  const paths = [];
  const client = new StateSet({
    fetch: async (url) => {
      paths.push(new URL(url).pathname);
      return jsonResponse({ data: [] });
    },
  });

  await client.chat.completions.create({ model: "m", messages: [] });
  await client.models.list();
  assert.deepEqual(paths, ["/v1/chat/completions", "/v1/models"]);
});

test("non-2xx responses preserve status, request id, and response body", async () => {
  const client = new StateSet({
    fetch: async () => jsonResponse(
      { detail: "model unavailable" },
      { status: 503, headers: { "x-request-id": "req_123" } },
    ),
  });

  await assert.rejects(
    () => client.models.list(),
    (error) => {
      assert.ok(error instanceof StateSetError);
      assert.equal(error.message, "model unavailable");
      assert.equal(error.status, 503);
      assert.equal(error.requestId, "req_123");
      assert.deepEqual(error.body, { detail: "model unavailable" });
      return true;
    },
  );
});

test("stream decodes SSE JSON and ignores the done marker", async () => {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode('data: {"delta":"hel"}\n\n'));
      controller.enqueue(encoder.encode('data: {"delta":"lo"}\n\ndata: [DONE]\n\n'));
      controller.close();
    },
  });
  const client = new StateSet({
    fetch: async (_url, init) => {
      assert.equal(JSON.parse(init.body).stream, true);
      return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
    },
  });

  const events = [];
  for await (const event of client.messages.stream({ model: "m", messages: [] })) {
    events.push(event);
  }
  assert.deepEqual(events, [{ delta: "hel" }, { delta: "lo" }]);
});

test("constructor rejects non-HTTP base URLs", () => {
  assert.throws(() => new StateSet({ baseURL: "file:///tmp/socket" }), /http/);
});

test("an already-aborted caller signal reaches fetch", async () => {
  const controller = new AbortController();
  controller.abort("cancelled");
  const client = new StateSet({
    fetch: async (_url, init) => {
      assert.equal(init.signal.aborted, true);
      throw new DOMException("aborted", "AbortError");
    },
  });

  await assert.rejects(
    () => client.health({ signal: controller.signal }),
    (error) => error instanceof StateSetError && error.message === "StateSet API request failed",
  );
});

test("request timeout produces a structured timeout error", async () => {
  const client = new StateSet({
    timeout: 5,
    fetch: async (_url, init) =>
      new Promise((_resolve, reject) => {
        init.signal.addEventListener(
          "abort",
          () => reject(new DOMException("aborted", "AbortError")),
          { once: true },
        );
      }),
  });

  await assert.rejects(
    () => client.health(),
    (error) =>
      error instanceof StateSetError &&
      error.message === "StateSet API request timed out after 5ms",
  );
});

test("studio client calls Weight Studio & SQLite store endpoints", async () => {
  const calls = [];
  const client = new StateSet({
    baseURL: "http://127.0.0.1:8765",
    apiKey: "studio-rw-token",
    fetch: async (url, init) => {
      calls.push({ url, method: init.method, auth: init.headers.get("authorization") });
      return jsonResponse({ ok: true });
    },
  });

  await client.studio.listModels();
  await client.studio.runAutopilot({ model_id: "preset:deepseek-r1-1.5b" });
  await client.studio.ingestTrace({ payload: { prompt: "hi", response: "hello" } });
  await client.studio.queryTraces({ failed_only: true, source: "otlp", limit: 10 });
  await client.studio.queryRollouts({ decision: "PROMOTE", limit: 5 });

  assert.equal(calls.length, 5);
  assert.equal(calls[0].url, "http://127.0.0.1:8765/api/weight-studio/models");
  assert.equal(calls[1].url, "http://127.0.0.1:8765/api/weight-studio/autopilot/run");
  assert.equal(calls[2].url, "http://127.0.0.1:8765/api/weight-studio/traces/ingest");
  assert.equal(
    calls[3].url,
    "http://127.0.0.1:8765/api/weight-studio/store/traces?failed_only=true&source=otlp&limit=10",
  );
  assert.equal(
    calls[4].url,
    "http://127.0.0.1:8765/api/weight-studio/store/rollouts?decision=PROMOTE&limit=5",
  );
  assert.equal(calls[0].auth, "Bearer studio-rw-token");
});

test("mcp HTTP client invokes JSON-RPC tools and surfaces JSON-RPC errors", async () => {
  const client = new StateSet({
    baseURL: "http://127.0.0.1:8765",
    apiKey: "ro-token",
    fetch: async (_url, init) => {
      const req = JSON.parse(init.body);
      if (req.method === "tools/call" && req.params.name === "studio_apply_edit") {
        return jsonResponse({
          jsonrpc: "2.0",
          id: req.id,
          error: { code: -32001, message: "Read-only token" },
        });
      }
      return jsonResponse({
        jsonrpc: "2.0",
        id: req.id,
        result: { tools: [{ name: "studio_list_models" }] },
      });
    },
  });

  const list = await client.mcp.listTools();
  assert.equal(list.tools[0].name, "studio_list_models");

  await assert.rejects(
    () => client.mcp.callTool("studio_apply_edit", { model_id: "m" }),
    (err) => err instanceof StateSetError && err.status === -32001,
  );
});

test("studio client calls Open-Weight RL Cockpit, Plan & Sandbox Pool endpoints", async () => {
  const calls = [];
  const client = new StateSet({
    baseURL: "http://127.0.0.1:8765",
    apiKey: "rl-token",
    fetch: async (url, init) => {
      calls.push({ url, method: init.method, body: init.body ? JSON.parse(init.body) : null });
      return jsonResponse({ ok: true });
    },
  });

  await client.studio.getRlTelemetry({ limit: 25, run_name: "gspo-1" });
  await client.studio.postRlTelemetry({ run_name: "gspo-1", step: 1, reward: 0.9 });
  await client.studio.controlRlRun({ action: "pause", reason: "guard" });
  await client.studio.planRlRun({ model: "Qwen/Qwen3-8B", algorithm: "dapo", cloud: "modal" });
  await client.studio.runSandboxPool({ domains: ["commerce", "coding", "sql"] });

  assert.equal(calls.length, 5);
  assert.equal(
    calls[0].url,
    "http://127.0.0.1:8765/api/weight-studio/rl/telemetry?limit=25&run_name=gspo-1",
  );
  assert.equal(calls[1].url, "http://127.0.0.1:8765/api/weight-studio/rl/telemetry");
  assert.equal(calls[2].url, "http://127.0.0.1:8765/api/weight-studio/rl/control");
  assert.equal(calls[2].body.action, "pause");
  assert.equal(calls[3].url, "http://127.0.0.1:8765/api/weight-studio/rl/plan");
  assert.equal(calls[4].url, "http://127.0.0.1:8765/api/weight-studio/rl/sandbox-pool");
});
