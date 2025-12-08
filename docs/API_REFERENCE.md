# StateSet Agents API Reference

## Overview

The StateSet Agents API provides a production-ready REST interface for training and deploying AI agents using reinforcement learning techniques (GRPO/GSPO).

**Base URL:** `https://api.stateset.ai/api/v1`

**API Version:** 2.0.0

## Authentication

All API endpoints require authentication using one of the following methods:

### API Key Authentication

Include your API key in the request headers:

```bash
curl -H "X-API-Key: your-api-key-here" https://api.stateset.ai/api/v1/health
```

### Bearer Token Authentication

Use a JWT bearer token in the Authorization header:

```bash
curl -H "Authorization: Bearer your-jwt-token" https://api.stateset.ai/api/v1/health
```

## Rate Limiting

- **Default limit:** 60 requests per minute
- **Rate limit headers** are included in all responses:
  - `X-RateLimit-Limit`: Maximum requests per window
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Unix timestamp when the limit resets

When rate limited, the API returns `429 Too Many Requests` with a `Retry-After` header.

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "prompts",
        "message": "At least one prompt is required",
        "code": "required"
      }
    ]
  },
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "path": "/api/v1/training"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `BAD_REQUEST` | 400 | Invalid request format |
| `UNAUTHORIZED` | 401 | Authentication required or failed |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

## Security

### Prompt Injection Protection

The API includes automatic detection and blocking of prompt injection attempts. Requests containing potentially harmful patterns will be rejected with a `400 Bad Request` response.

**Blocked patterns include:**
- Instruction override attempts ("ignore previous instructions")
- Role manipulation ("you are now in developer mode")
- System prompt extraction attempts
- Delimiter-based injection attacks
- Jailbreak attempts

### Security Headers

All responses include the following security headers:

| Header | Value |
|--------|-------|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `X-XSS-Protection` | `1; mode=block` |
| `Content-Security-Policy` | `default-src 'self'...` |
| `X-Permitted-Cross-Domain-Policies` | `none` |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` (production) |

### Authentication Lockout

After 5 failed authentication attempts, the client IP is temporarily locked out for 5 minutes.

---

## Endpoints

### Health Check

Check the health status of the API.

```
GET /api/v1/health
```

**Authentication:** Not required

**Response:**

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime_seconds": 3600.5,
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 5.2
    }
  ],
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

---

### Agents

#### Create Agent

Create a new AI agent with custom configuration.

```
POST /api/v1/agents
```

**Request Body:**

```json
{
  "model_name": "gpt2",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "system_prompt": "You are a helpful assistant.",
  "use_chat_template": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_name` | string | Yes | Model name or path |
| `max_new_tokens` | integer | No | Maximum tokens to generate (1-4096, default: 512) |
| `temperature` | float | No | Sampling temperature (0.0-2.0, default: 0.8) |
| `top_p` | float | No | Top-p sampling (0.0-1.0, default: 0.9) |
| `top_k` | integer | No | Top-k sampling (1-1000, default: 50) |
| `system_prompt` | string | No | System prompt for the agent |
| `use_chat_template` | boolean | No | Use chat template (default: true) |

**Response (201 Created):**

```json
{
  "agent_id": "agent_550e8400",
  "created_at": "2024-01-15T10:30:00.000Z",
  "config": {
    "model_name": "gpt2",
    "max_new_tokens": 256,
    "temperature": 0.7
  },
  "message": "Agent created successfully"
}
```

#### List Agents

Get a paginated list of all agents.

```
GET /api/v1/agents?page=1&page_size=20&status=active
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `page` | integer | Page number (default: 1) |
| `page_size` | integer | Items per page (1-100, default: 20) |
| `status` | string | Filter by status (active, inactive) |

**Response:**

```json
{
  "items": [
    {
      "agent_id": "agent_550e8400",
      "model_name": "gpt2",
      "created_at": "2024-01-15T10:30:00.000Z",
      "conversation_count": 42,
      "total_tokens_used": 15000,
      "status": "active"
    }
  ],
  "total": 5,
  "page": 1,
  "page_size": 20,
  "has_next": false,
  "has_prev": false
}
```

#### Get Agent

Get details of a specific agent.

```
GET /api/v1/agents/{agent_id}
```

**Response:**

```json
{
  "agent_id": "agent_550e8400",
  "model_name": "gpt2",
  "created_at": "2024-01-15T10:30:00.000Z",
  "conversation_count": 42,
  "total_tokens_used": 15000,
  "config": {
    "model_name": "gpt2",
    "max_new_tokens": 256,
    "temperature": 0.7
  },
  "status": "active"
}
```

#### Delete Agent

Delete an agent and all associated conversations.

```
DELETE /api/v1/agents/{agent_id}
```

**Response:** `204 No Content`

---

### Training

#### Start Training Job

Start a new GRPO/GSPO training job.

```
POST /api/v1/training
```

**Request Body:**

```json
{
  "prompts": ["What is machine learning?", "Explain neural networks"],
  "strategy": "computational",
  "num_iterations": 10,
  "use_neural_rewards": true,
  "use_ruler_rewards": false,
  "idempotency_key": "unique-request-id"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompts` | array[string] | Yes | Training prompts (1-8 prompts, max 4000 chars each) |
| `strategy` | string | No | Training strategy: `computational`, `distributed`, `grpo`, `gspo` (default: `computational`) |
| `num_iterations` | integer | No | Number of training iterations (1-50, default: 1) |
| `use_neural_rewards` | boolean | No | Enable neural reward models (default: true) |
| `use_ruler_rewards` | boolean | No | Enable RULER LLM judges (default: false) |
| `idempotency_key` | string | No | Unique key for request deduplication |

**Response (202 Accepted):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "starting",
  "strategy": "computational",
  "metrics": {
    "iterations_completed": 0,
    "total_trajectories": 0,
    "average_reward": 0.0
  },
  "started_at": "2024-01-15T10:30:00.000Z",
  "request_id": "req_abc123",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Get Training Status

Get the status of a training job.

```
GET /api/v1/training/{job_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Training job ID |

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "strategy": "computational",
  "metrics": {
    "iterations_completed": 5,
    "total_trajectories": 500,
    "average_reward": 0.75,
    "computation_used": 150.5
  },
  "started_at": "2024-01-15T10:30:00.000Z",
  "timestamp": "2024-01-15T10:35:00.000Z"
}
```

**Status Values:**

| Status | Description |
|--------|-------------|
| `pending` | Job is queued |
| `starting` | Job is initializing |
| `running` | Job is actively training |
| `completed` | Job finished successfully |
| `failed` | Job failed with error |
| `cancelled` | Job was cancelled |

#### Cancel Training Job

Cancel a running training job.

```
DELETE /api/v1/training/{job_id}
```

**Response:**

```json
{
  "message": "Training job cancelled",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "request_id": "req_abc123"
}
```

---

### Conversations

#### Create/Continue Conversation

Send a message to an agent and get a response.

```
POST /api/v1/conversations
```

**Request Body (Single Message):**

```json
{
  "message": "Hello, how can you help me?",
  "conversation_id": "conv_abc123",
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

**Request Body (Multi-turn):**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help?"},
    {"role": "user", "content": "What's the weather like?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | No* | User message (single-turn) |
| `messages` | array | No* | Full conversation history (multi-turn) |
| `conversation_id` | string | No | Existing conversation ID to continue |
| `max_tokens` | integer | No | Maximum response tokens (1-4096, default: 512) |
| `temperature` | float | No | Sampling temperature (0.0-2.0, default: 0.8) |
| `stream` | boolean | No | Enable streaming response (default: false) |

*Either `message` or `messages` is required, but not both.

**Response:**

```json
{
  "conversation_id": "conv_abc123",
  "response": "Hello! I'm here to help you with any questions you have.",
  "tokens_used": 15,
  "processing_time_ms": 234.5,
  "context": {
    "turn_count": 1,
    "strategy": "default"
  },
  "metadata": {
    "model": "stateset-agent-v2"
  },
  "request_id": "req_abc123",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Get Conversation History

Get conversation details and message history.

```
GET /api/v1/conversations/{conversation_id}
```

**Response:**

```json
{
  "conversation_id": "conv_abc123",
  "user_id": "user_123",
  "created_at": "2024-01-15T10:30:00.000Z",
  "message_count": 4,
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help?"}
  ],
  "request_id": "req_abc123"
}
```

#### End Conversation

End a conversation and clean up resources.

```
DELETE /api/v1/conversations/{conversation_id}
```

**Response:**

```json
{
  "message": "Conversation ended",
  "conversation_id": "conv_abc123",
  "total_messages": 4,
  "request_id": "req_abc123"
}
```

---

### Metrics

Get system and API metrics (admin only).

```
GET /api/v1/metrics
```

**Required Role:** `admin`

**Response:**

```json
{
  "system": {
    "uptime_seconds": 3600.5,
    "active_services": 3,
    "active_jobs": 5,
    "active_conversations": 10
  },
  "api": {
    "total_requests": 10000,
    "requests_by_endpoint": {
      "POST:/api/v1/conversations": 5000,
      "GET:/api/v1/health": 3000
    },
    "status_codes": {
      "200": 9500,
      "400": 300,
      "500": 200
    },
    "latency": {
      "avg_ms": 45.2,
      "p50_ms": 30.0,
      "p95_ms": 120.0,
      "p99_ms": 250.0
    }
  },
  "training": {
    "total_jobs": 100,
    "running_jobs": 5,
    "completed_jobs": 90,
    "failed_jobs": 5
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

---

### Scaling

Scale computational resources (admin only).

```
POST /api/v1/scale
```

**Required Role:** `admin`

**Request Body:**

```json
{
  "scale_factor": 2.0,
  "apply_to_all": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scale_factor` | float | Yes | Scaling factor (0.1-10.0) |
| `apply_to_all` | boolean | No | Apply to all engines (default: false) |
| `target_engines` | array[string] | No | Specific engine IDs to scale |

**Response:**

```json
{
  "scale_factor": 2.0,
  "results": {
    "engine_1": {"status": "scaled"},
    "engine_2": {"status": "scaled"}
  },
  "message": "Scaling completed with factor 2.0",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

---

## WebSocket API

Connect to the WebSocket endpoint for real-time interactions.

```
WebSocket /api/v1/ws
```

### Authentication

Include API key in connection headers:

```javascript
const ws = new WebSocket('wss://api.stateset.ai/api/v1/ws', {
  headers: {
    'X-API-Key': 'your-api-key'
  }
});
```

### Message Types

#### Ping/Pong

```json
// Send
{"type": "ping"}

// Receive
{"type": "pong", "timestamp": "2024-01-15T10:30:00.000Z"}
```

#### Chat

```json
// Send
{
  "type": "chat",
  "message": "Hello!"
}

// Receive
{
  "type": "chat_response",
  "response": "Hi there! How can I help?",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Subscribe to Job Updates

```json
// Send
{
  "type": "subscribe_job",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}

// Receive
{
  "type": "job_update",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

---

## SDK Examples

### Python

```python
import httpx

API_KEY = "your-api-key"
BASE_URL = "https://api.stateset.ai/api/v1"

headers = {"X-API-Key": API_KEY}

# Start training
response = httpx.post(
    f"{BASE_URL}/training",
    headers=headers,
    json={
        "prompts": ["What is machine learning?"],
        "num_iterations": 10
    }
)
job = response.json()
print(f"Started job: {job['job_id']}")

# Chat with agent
response = httpx.post(
    f"{BASE_URL}/conversations",
    headers=headers,
    json={"message": "Hello!"}
)
print(response.json()["response"])
```

### JavaScript/TypeScript

```typescript
const API_KEY = 'your-api-key';
const BASE_URL = 'https://api.stateset.ai/api/v1';

// Start training
const trainingResponse = await fetch(`${BASE_URL}/training`, {
  method: 'POST',
  headers: {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    prompts: ['What is machine learning?'],
    num_iterations: 10
  })
});
const job = await trainingResponse.json();
console.log(`Started job: ${job.job_id}`);

// Chat with agent
const chatResponse = await fetch(`${BASE_URL}/conversations`, {
  method: 'POST',
  headers: {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ message: 'Hello!' })
});
const chat = await chatResponse.json();
console.log(chat.response);
```

### cURL

```bash
# Health check
curl https://api.stateset.ai/api/v1/health

# Start training
curl -X POST https://api.stateset.ai/api/v1/training \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["What is ML?"], "num_iterations": 10}'

# Chat
curl -X POST https://api.stateset.ai/api/v1/conversations \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

---

## Changelog

### v2.0.0 (Current)

- Complete API redesign with versioned endpoints
- Unified error response format
- Security headers on all responses
- Rate limiting with sliding window algorithm
- OpenTelemetry tracing support
- Comprehensive input validation
- WebSocket support for real-time updates
- Idempotency key support for training requests

### v1.0.0 (Legacy)

- Initial API release
- Basic training and conversation endpoints
