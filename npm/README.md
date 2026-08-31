# `@stateset/agents`

Typed, dependency-free Node.js client for the StateSet Agents API. It supports
the Anthropic-style Messages API, OpenAI-compatible chat completions and model
listing, server-sent-event streaming, and health checks.

```bash
npm install @stateset/agents
```

```js
import { StateSet } from "@stateset/agents";

const stateset = new StateSet({
  baseURL: "https://agents.example.com",
  apiKey: process.env.STATESET_API_KEY,
});

const response = await stateset.messages.create({
  model: "Qwen/Qwen3.8-27B",
  messages: [{ role: "user", content: "Where is order 77701?" }],
  max_tokens: 200,
});

console.log(response.content);
```

OpenAI-compatible calls use the familiar resource shape:

```js
const completion = await stateset.chat.completions.create({
  model: "Qwen/Qwen3.8-27B",
  messages: [{ role: "user", content: "Hello" }],
});
```

Streaming methods return an async iterator of decoded SSE JSON payloads:

```js
for await (const event of stateset.chat.completions.stream({
  model: "Qwen/Qwen3.8-27B",
  messages: [{ role: "user", content: "Hello" }],
})) {
  process.stdout.write(event.choices?.[0]?.delta?.content ?? "");
}
```

Requires Node.js 18 or newer. Pass a custom `fetch` implementation to the
constructor for compatible runtimes, testing, proxies, or instrumentation.
