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

const error: StateSetError = new StateSetError("test");
void error.status;
