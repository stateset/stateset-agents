export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface Message {
  role: MessageRole;
  content?: string | Array<Record<string, unknown>> | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface MessagesRequest {
  model: string;
  messages: Message[];
  system?: string | Array<Record<string, unknown>>;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop_sequences?: string[];
  stream?: boolean;
  tools?: Array<Record<string, unknown>>;
  tool_choice?: string | Record<string, unknown>;
  metadata?: Record<string, unknown>;
  response_format?: "anthropic" | "openai";
  [key: string]: unknown;
}

export interface ChatCompletionRequest {
  model: string;
  messages: Message[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
  stop?: string | string[];
  tools?: Array<Record<string, unknown>>;
  tool_choice?: string | Record<string, unknown>;
  [key: string]: unknown;
}

export interface RequestOptions {
  headers?: HeadersInit;
  signal?: AbortSignal;
  timeout?: number;
}

export interface StateSetOptions {
  baseURL?: string;
  apiKey?: string;
  timeout?: number;
  fetch?: typeof fetch;
}

export class StateSetError extends Error {
  status?: number;
  requestId?: string;
  body?: unknown;
  cause?: unknown;
}

export class StateSet {
  constructor(options?: StateSetOptions);
  readonly baseURL: string;
  readonly apiKey?: string;
  readonly timeout: number;

  readonly messages: {
    create<T = Record<string, unknown>>(body: MessagesRequest, options?: RequestOptions): Promise<T>;
    stream<T = Record<string, unknown>>(body: MessagesRequest, options?: RequestOptions): AsyncGenerator<T>;
  };
  readonly chat: {
    completions: {
      create<T = Record<string, unknown>>(body: ChatCompletionRequest, options?: RequestOptions): Promise<T>;
      stream<T = Record<string, unknown>>(body: ChatCompletionRequest, options?: RequestOptions): AsyncGenerator<T>;
    };
  };
  readonly models: {
    list<T = Record<string, unknown>>(options?: RequestOptions): Promise<T>;
  };

  request<T = unknown>(path: string, options?: RequestOptions & { method?: string; body?: unknown }): Promise<T>;
  stream<T = Record<string, unknown>>(path: string, options?: RequestOptions & { body?: unknown }): AsyncGenerator<T>;
  health<T = Record<string, unknown>>(options?: RequestOptions): Promise<T>;
}

export default StateSet;
