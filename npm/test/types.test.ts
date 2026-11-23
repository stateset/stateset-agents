import StateSet, {
  type ChatCompletionRequest,
  type MessagesRequest,
  StateSetError,
} from "@stateset/agents";

const client = new StateSet({
  baseURL: "https://agents.example.com",
  apiKey: "test-key",
  timeout: 1_000,
});

const message: MessagesRequest = {
  model: "Qwen/Qwen3.8-27B",
  messages: [{ role: "user", content: "hello" }],
};

const completion: ChatCompletionRequest = {
  model: "Qwen/Qwen3.8-27B",
  messages: [{ role: "user", content: "hello" }],
};

void client.messages.create(message);
void client.messages.stream(message);
void client.chat.completions.create(completion);
void client.chat.completions.stream(completion);
void client.models.list();
void client.health();
void client.studio.listModels();
void client.studio.runAutopilot({ model_id: "preset:deepseek-r1-1.5b" });
void client.studio.queryTraces({ failed_only: true, limit: 20 });
void client.studio.queryRollouts({ decision: "PROMOTE" });
void client.studio.getRlTelemetry({ limit: 40 });
void client.studio.postRlTelemetry({ step: 1, reward: 0.85 });
void client.studio.controlRlRun({ action: "rollback", reason: "hacking guard" });
void client.studio.planRlRun({ model: "Qwen/Qwen3-8B", algorithm: "gspo", cloud: "modal" });
void client.studio.runSandboxPool({ domains: ["commerce", "coding", "sql"] });
void client.mcp.listTools();
void client.mcp.callTool("studio_list_models", {});
void client.mcp.readResource("stateset://studio/traces");

const error: StateSetError = new StateSetError("test");
void error.status;
