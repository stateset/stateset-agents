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

export interface TraceQueryParams {
  failed_only?: boolean;
  source?: string;
  search?: string;
  labeled_only?: boolean;
  limit?: number;
}

export interface RolloutQueryParams {
  decision?: "PROMOTE" | "HOLD_CANARY" | "ROLLBACK" | string;
  model_id?: string;
  stage?: string;
  limit?: number;
}

export interface RlControlRequest {
  action: "pause" | "resume" | "rollback";
  run_name?: string;
  reason?: string;
}

export interface RlPlanRequest {
  model?: string;
  algorithm?: "grpo" | "gspo" | "dapo" | "vapo" | "gepo" | "cispo";
  cloud?: "local" | "runpod" | "modal" | "k8s" | "river" | "colab";
  gpus?: number;
  chunked_loss_chunk_size?: number;
  live_vllm_endpoint?: string;
  [key: string]: unknown;
}

export interface SandboxPoolRequest {
  domains?: Array<"commerce" | "coding" | "sql" | "rest_api" | "docker_code" | "web_agent">;
  include_failure_pairs?: boolean;
  max_concurrency?: number;
}

export interface RlTrajectoryQueryParams {
  run_name?: string;
  domain?: string;
  failed_only?: boolean;
  limit?: number;
}

export class WeightStudioClient {
  constructor(client: StateSet);
  listModels<T = Record<string, unknown>>(options?: RequestOptions): Promise<T>;
  autoHeal<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  runRedTeam<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  runTournament<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  runCanaryGate<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  runAutopilot<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  loadLoraAdapter<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  compareLiveEndpoints<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  ingestTrace<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  tracesSummary<T = Record<string, unknown>>(options?: RequestOptions): Promise<T>;
  auditRewardCalibration<T = Record<string, unknown>>(body?: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  healFromLiveTraces<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  queryTraces<T = Record<string, unknown>>(params?: TraceQueryParams, options?: RequestOptions): Promise<T>;
  queryRollouts<T = Record<string, unknown>>(params?: RolloutQueryParams, options?: RequestOptions): Promise<T>;
  storeStats<T = Record<string, unknown>>(options?: RequestOptions): Promise<T>;
  getRlTelemetry<T = Record<string, unknown>>(params?: { limit?: number; run_name?: string }, options?: RequestOptions): Promise<T>;
  postRlTelemetry<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  controlRlRun<T = Record<string, unknown>>(body: RlControlRequest, options?: RequestOptions): Promise<T>;
  planRlRun<T = Record<string, unknown>>(body?: RlPlanRequest, options?: RequestOptions): Promise<T>;
  runSandboxPool<T = Record<string, unknown>>(body?: SandboxPoolRequest, options?: RequestOptions): Promise<T>;
  recordRlTrajectory<T = Record<string, unknown>>(body: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  queryRlTrajectories<T = Record<string, unknown>>(params?: RlTrajectoryQueryParams, options?: RequestOptions): Promise<T>;
}

export class McpHttpClient {
  constructor(client: StateSet, path?: string);
  rpc<T = Record<string, unknown>>(method: string, params?: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  initialize<T = Record<string, unknown>>(options?: RequestOptions): Promise<T>;
  listTools<T = Record<string, unknown>>(options?: RequestOptions): Promise<T>;
  callTool<T = Record<string, unknown>>(name: string, args?: Record<string, unknown>, options?: RequestOptions): Promise<T>;
  listResources<T = Record<string, unknown>>(options?: RequestOptions): Promise<T>;
  readResource<T = Record<string, unknown>>(uri: string, options?: RequestOptions): Promise<T>;
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
  readonly studio: WeightStudioClient;
  readonly mcp: McpHttpClient;

  request<T = unknown>(path: string, options?: RequestOptions & { method?: string; body?: unknown }): Promise<T>;
  stream<T = Record<string, unknown>>(path: string, options?: RequestOptions & { body?: unknown }): AsyncGenerator<T>;
  health<T = Record<string, unknown>>(options?: RequestOptions): Promise<T>;
}

export default StateSet;
