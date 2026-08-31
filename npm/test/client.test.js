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
