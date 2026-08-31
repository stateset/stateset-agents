const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_TIMEOUT_MS = 30_000;

/** Error returned for a non-2xx StateSet API response. */
export class StateSetError extends Error {
  constructor(message, { status, requestId, body, cause } = {}) {
    super(message, { cause });
    this.name = "StateSetError";
    this.status = status;
    this.requestId = requestId;
    this.body = body;
  }
}

function cleanBaseURL(value) {
  const url = String(value || DEFAULT_BASE_URL).replace(/\/+$/, "");
  if (!/^https?:\/\//i.test(url)) {
    throw new TypeError("baseURL must use http:// or https://");
  }
  return url;
}

async function responseBody(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  const text = await response.text();
  return text || null;
}

function errorMessage(status, body) {
  if (body && typeof body === "object") {
    const detail = body.detail ?? body.error?.message ?? body.error;
    if (typeof detail === "string" && detail) return detail;
  }
  if (typeof body === "string" && body) return body;
  return `StateSet API request failed with status ${status}`;
}

function linkAbortSignal(signal, controller) {
  const abort = () => controller.abort(signal?.reason);
  if (signal?.aborted) abort();
  else signal?.addEventListener("abort", abort, { once: true });
  return () => signal?.removeEventListener("abort", abort);
}

async function* parseSSE(response) {
  if (!response.body) {
    throw new StateSetError("Streaming response has no body", {
      status: response.status,
      requestId: response.headers.get("x-request-id") || undefined,
    });
  }

  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true }).replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data || data === "[DONE]") continue;
      try {
        yield JSON.parse(data);
      } catch (cause) {
        throw new StateSetError("StateSet API returned invalid SSE JSON", {
          status: response.status,
          requestId: response.headers.get("x-request-id") || undefined,
          body: data,
          cause,
        });
      }
    }
  }
}

/** Typed client for a StateSet Agents API deployment. */
export class StateSet {
  constructor({
    baseURL = DEFAULT_BASE_URL,
    apiKey,
    timeout = DEFAULT_TIMEOUT_MS,
    fetch: fetchImpl,
  } = {}) {
    this.baseURL = cleanBaseURL(baseURL);
    this.apiKey = apiKey;
    this.timeout = timeout;
    this.fetch = fetchImpl ?? globalThis.fetch;
    if (typeof this.fetch !== "function") {
      throw new TypeError("A Fetch API implementation is required");
    }

    this.messages = {
      create: (body, options) =>
        this.request("/v1/messages", { method: "POST", body, ...options }),
      stream: (body, options) =>
        this.stream("/v1/messages", {
          body: { ...body, stream: true },
          ...options,
        }),
    };
    this.chat = {
      completions: {
        create: (body, options) =>
          this.request("/v1/chat/completions", {
            method: "POST",
            body,
            ...options,
          }),
        stream: (body, options) =>
          this.stream("/v1/chat/completions", {
            body: { ...body, stream: true },
            ...options,
          }),
      },
    };
    this.models = { list: (options) => this.request("/v1/models", options) };
  }

  async request(
    path,
    { method = "GET", body, headers = {}, signal, timeout = this.timeout } = {},
  ) {
    const controller = new AbortController();
    const timer =
      timeout > 0 ? setTimeout(() => controller.abort(), timeout) : undefined;
    const unlinkAbortSignal = linkAbortSignal(signal, controller);
    const requestHeaders = new Headers(headers);
    requestHeaders.set("accept", "application/json");
    if (body !== undefined) requestHeaders.set("content-type", "application/json");
    if (this.apiKey) requestHeaders.set("authorization", `Bearer ${this.apiKey}`);

    try {
      const response = await this.fetch(`${this.baseURL}${path}`, {
        method,
        headers: requestHeaders,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      const parsed = await responseBody(response);
      if (!response.ok) {
        throw new StateSetError(errorMessage(response.status, parsed), {
          status: response.status,
          requestId: response.headers.get("x-request-id") || undefined,
          body: parsed,
        });
      }
      return parsed;
    } catch (cause) {
      if (cause instanceof StateSetError) throw cause;
      const timedOut = controller.signal.aborted && !signal?.aborted;
      throw new StateSetError(
        timedOut
          ? `StateSet API request timed out after ${timeout}ms`
          : "StateSet API request failed",
        { cause },
      );
    } finally {
      if (timer) clearTimeout(timer);
      unlinkAbortSignal();
    }
  }

  async *stream(
    path,
    { body, headers = {}, signal, timeout = this.timeout } = {},
  ) {
    const controller = new AbortController();
    const timer =
      timeout > 0 ? setTimeout(() => controller.abort(), timeout) : undefined;
    const unlinkAbortSignal = linkAbortSignal(signal, controller);
    const requestHeaders = new Headers(headers);
    requestHeaders.set("accept", "text/event-stream");
    requestHeaders.set("content-type", "application/json");
    if (this.apiKey) requestHeaders.set("authorization", `Bearer ${this.apiKey}`);

    try {
      const response = await this.fetch(`${this.baseURL}${path}`, {
        method: "POST",
        headers: requestHeaders,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok) {
        const parsed = await responseBody(response);
        throw new StateSetError(errorMessage(response.status, parsed), {
          status: response.status,
          requestId: response.headers.get("x-request-id") || undefined,
          body: parsed,
        });
      }
      yield* parseSSE(response);
    } catch (cause) {
      if (cause instanceof StateSetError) throw cause;
      const timedOut = controller.signal.aborted && !signal?.aborted;
      throw new StateSetError(
        timedOut
          ? `StateSet API stream timed out after ${timeout}ms`
          : "StateSet API stream failed",
        { cause },
      );
    } finally {
      if (timer) clearTimeout(timer);
      unlinkAbortSignal();
      controller.abort();
    }
  }

  health(options) {
    return this.request("/health", options);
  }
}

export default StateSet;
