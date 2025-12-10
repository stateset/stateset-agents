# StateSet Agents API Examples

This directory contains example code demonstrating how to interact with the StateSet Agents API as a client.

## Available Examples

### 1. `api_client_example.py` - Comprehensive Async Client

A full-featured, production-ready async client using `httpx` that demonstrates all major API features.

**Features:**
- Health checks and API information
- Agent creation and management
- Multi-turn conversations
- Agent and conversation listing
- System metrics retrieval
- Proper error handling and context managers

**Prerequisites:**
```bash
pip install httpx python-dotenv
```

**Usage:**
```bash
# Start the API server
python -m api.main

# In another terminal, run the example
python examples/api_client_example.py
```

### 2. `api_client_simple.py` - Simple Synchronous Client

A simplified synchronous client using `requests` for quick prototyping and integration into existing synchronous codebases.

**Features:**
- Basic health checks
- Agent creation
- Simple conversations
- Conversation listing
- Easy to understand and modify

**Prerequisites:**
```bash
pip install requests python-dotenv
```

**Usage:**
```bash
# Start the API server
python -m api.main

# In another terminal, run the example
python examples/api_client_simple.py
```

### 3. `interactive_chatbot.py` - Interactive CLI Chatbot

A fully-featured interactive command-line chatbot demonstrating real-world API usage. This example shows how to build a production-ready conversational application.

**Features:**
- Interactive terminal UI with rich formatting (optional)
- Multi-turn conversation with context
- Command system (/help, /clear, /history)
- Error handling and reconnection
- Customizable system prompts
- Works with and without rich terminal formatting

**Prerequisites:**
```bash
pip install requests python-dotenv
# Optional for enhanced formatting:
pip install rich
```

**Usage:**
```bash
# Start the API server
python -m api.main

# Run the chatbot with default prompt
python examples/interactive_chatbot.py

# Run with custom system prompt
python examples/interactive_chatbot.py "You are a Python programming expert"
```

## API Server Setup

Before running any examples, you need to start the StateSet Agents API server.

### Method 1: Using the module directly
```bash
python -m api.main
```

### Method 2: Using the CLI
```bash
stateset-agents serve --host 0.0.0.0 --port 8000
```

### Method 3: Using Docker
```bash
docker build -t stateset/agents:latest -f deployment/docker/Dockerfile .
docker run -p 8000:8000 stateset/agents:latest
```

The API will be available at `http://localhost:8000`

## Environment Variables

Create a `.env` file in the root directory:

```bash
# API Configuration
API_BASE_URL=http://localhost:8000
API_JWT_SECRET=test_secret_for_development_only_not_for_prod
API_ENV=development

# Optional: Model configuration
MODEL_CACHE_DIR=/path/to/model/cache
```

## Key API Endpoints

### Health & Info
- `GET /` - API information and available endpoints
- `GET /health` - Health check with component status
- `GET /ready` - Readiness probe (for Kubernetes)
- `GET /live` - Liveness probe (for Kubernetes)

### Agent Management
- `POST /agents` - Create a new agent
- `GET /agents` - List all agents (with pagination)
- `GET /agents/{agent_id}` - Get agent details
- `DELETE /agents/{agent_id}` - Delete an agent

### Conversations
- `POST /conversations` - Send a message and get a response
- `GET /conversations` - List all conversations (with pagination)
- `GET /conversations/{conversation_id}` - Get conversation history
- `DELETE /conversations/{conversation_id}` - Delete a conversation

### Training (Advanced)
- `POST /training/jobs` - Start a training job
- `GET /training/jobs` - List training jobs
- `GET /training/jobs/{job_id}` - Get job status
- `DELETE /training/jobs/{job_id}` - Cancel a job

### Metrics & Monitoring
- `GET /metrics/summary` - System metrics overview
- `GET /metrics/prometheus` - Prometheus-formatted metrics
- `GET /circuits` - Circuit breaker status

## Quick Start Code Snippets

### Creating an Agent

**Async (httpx):**
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/agents",
        json={
            "model_name": "gpt2",
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.7,
            "max_new_tokens": 256
        },
        headers={"Authorization": "Bearer YOUR_API_KEY"}
    )
    agent = response.json()
    print(f"Agent ID: {agent['agent_id']}")
```

**Sync (requests):**
```python
import requests

response = requests.post(
    "http://localhost:8000/agents",
    json={
        "model_name": "gpt2",
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_new_tokens": 256
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)
agent = response.json()
print(f"Agent ID: {agent['agent_id']}")
```

### Sending a Message

**Async (httpx):**
```python
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/conversations",
        json={
            "messages": [
                {"role": "user", "content": "Hello! How are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 256
        },
        headers={"Authorization": "Bearer YOUR_API_KEY"}
    )
    result = response.json()
    print(f"Agent: {result['response']}")
    print(f"Conversation ID: {result['conversation_id']}")
```

**Sync (requests):**
```python
response = requests.post(
    "http://localhost:8000/conversations",
    json={
        "messages": [
            {"role": "user", "content": "Hello! How are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)
result = response.json()
print(f"Agent: {result['response']}")
print(f"Conversation ID: {result['conversation_id']}")
```

### Multi-Turn Conversation

```python
messages = [
    {"role": "user", "content": "What is reinforcement learning?"}
]

# First turn
response = requests.post(
    "http://localhost:8000/conversations",
    json={"messages": messages},
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)
result = response.json()
conversation_id = result["conversation_id"]

# Add agent's response to history
messages.append({"role": "assistant", "content": result["response"]})

# Second turn
messages.append({"role": "user", "content": "Can you give an example?"})

response = requests.post(
    "http://localhost:8000/conversations",
    json={
        "messages": messages,
        "conversation_id": conversation_id
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)
result = response.json()
print(result["response"])
```

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `201` - Created (for POST requests)
- `204` - No Content (for DELETE requests)
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (missing/invalid API key)
- `404` - Not Found (resource doesn't exist)
- `422` - Unprocessable Entity (validation error)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

Error responses include a JSON body:
```json
{
  "error": "ValidationError",
  "message": "Message content cannot be empty",
  "details": {
    "field": "messages[0].content"
  },
  "timestamp": 1234567890.123
}
```

## Authentication

The API uses JWT bearer token authentication. Include your API key in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

For development, you can use the default key from your `.env` file:
```
API_JWT_SECRET=test_secret_for_development_only_not_for_prod
```

**Important:** Never use the development secret in production!

## API Documentation

Once the server is running, visit these URLs for interactive documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/openapi.json

## Additional Resources

- [Main README](../README.md) - Project overview
- [Quick Start Guide](../QUICKSTART.md) - Getting started with StateSet Agents
- [API Source Code](../api/) - API implementation details
- [Training Examples](.) - Examples for training agents

## Support

- Documentation: https://stateset-agents.readthedocs.io/
- Discord: https://discord.gg/stateset
- Issues: https://github.com/stateset/stateset-agents/issues

## License

See [LICENSE](../LICENSE) for details.
