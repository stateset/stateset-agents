# API Request/Response Examples

Comprehensive examples for all StateSet Agents API endpoints with realistic request/response payloads.

## Table of Contents

- [Authentication](#authentication)
- [Agent Management](#agent-management)
- [Conversations](#conversations)
- [Training](#training)
- [Health & Monitoring](#health--monitoring)
- [Batch Operations](#batch-operations)
- [Error Handling](#error-handling)

---

## Authentication

The API supports two authentication methods: API keys and JWT tokens.

### API Key Authentication

Include your API key in the `X-API-Key` header or as a Bearer token.

#### Example: Using X-API-Key Header

```bash
curl -X GET https://api.stateset.io/agents \
  -H "X-API-Key: your-api-key-here"
```

```python
import requests

headers = {
    "X-API-Key": "your-api-key-here"
}

response = requests.get(
    "https://api.stateset.io/agents",
    headers=headers
)
print(response.json())
```

#### Example: Using Bearer Token

```bash
curl -X GET https://api.stateset.io/agents \
  -H "Authorization: Bearer your-api-key-here"
```

```python
import requests

headers = {
    "Authorization": "Bearer your-api-key-here"
}

response = requests.get(
    "https://api.stateset.io/agents",
    headers=headers
)
print(response.json())
```

### JWT Token Authentication

Generate a JWT token and use it for authenticated requests.

#### Request: Generate Token

```bash
curl -X POST https://api.stateset.io/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_12345",
    "roles": ["user", "trainer"]
  }'
```

#### Response: Token Generated

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyXzEyMzQ1Iiwicm9sZXMiOlsidXNlciIsInRyYWluZXIiXSwiaWF0IjoxNzA1MzE0MDAwLCJleHAiOjE3MDUzMTc2MDAsImp0aSI6ImFiYzEyM3h5ejc4OSJ9.signature",
  "expires_at": "2024-01-15T11:30:00.000Z",
  "user_id": "user_12345",
  "roles": ["user", "trainer"]
}
```

#### Using JWT Token

```bash
curl -X GET https://api.stateset.io/agents \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

```python
import requests

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

headers = {
    "Authorization": f"Bearer {token}"
}

response = requests.get(
    "https://api.stateset.io/agents",
    headers=headers
)
print(response.json())
```

---

## Agent Management

### Create Agent

Create a new AI agent with custom configuration.

#### Request

```bash
curl -X POST https://api.stateset.io/agents \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "system_prompt": "You are a helpful customer support assistant for an e-commerce platform.",
    "use_chat_template": true
  }'
```

```python
import requests

url = "https://api.stateset.io/agents"
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

payload = {
    "model_name": "gpt2",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "system_prompt": "You are a helpful customer support assistant for an e-commerce platform.",
    "use_chat_template": True
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

#### Response (201 Created)

```json
{
  "agent_id": "agent_550e8400e29b41d4",
  "created_at": "2024-01-15T10:30:00.000Z",
  "config": {
    "model_name": "gpt2",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "system_prompt": "You are a helpful customer support assistant for an e-commerce platform.",
    "use_chat_template": true
  },
  "message": "Agent created successfully"
}
```

### List Agents

Get a paginated list of all agents.

#### Request

```bash
curl -X GET "https://api.stateset.io/agents?page=1&page_size=20&status=active" \
  -H "Authorization: Bearer your-api-key"
```

```python
import requests

url = "https://api.stateset.io/agents"
headers = {"Authorization": "Bearer your-api-key"}
params = {
    "page": 1,
    "page_size": 20,
    "status": "active"
}

response = requests.get(url, headers=headers, params=params)
print(response.json())
```

#### Response (200 OK)

```json
{
  "items": [
    {
      "agent_id": "agent_550e8400e29b41d4",
      "model_name": "gpt2",
      "created_at": "2024-01-15T10:30:00.000Z",
      "conversation_count": 42,
      "total_tokens_used": 15234,
      "config": {
        "temperature": 0.7,
        "max_new_tokens": 256
      },
      "status": "active"
    },
    {
      "agent_id": "agent_6629f511a37c52e5",
      "model_name": "gpt2",
      "created_at": "2024-01-14T15:22:00.000Z",
      "conversation_count": 18,
      "total_tokens_used": 8921,
      "config": {
        "temperature": 0.8,
        "max_new_tokens": 512
      },
      "status": "active"
    }
  ],
  "total": 2,
  "page": 1,
  "page_size": 20,
  "has_next": false,
  "has_prev": false,
  "request_id": "req_abc123xyz789",
  "timestamp": "2024-01-15T10:35:00.000Z"
}
```

### Get Agent Details

Get detailed information about a specific agent.

#### Request

```bash
curl -X GET https://api.stateset.io/agents/agent_550e8400e29b41d4 \
  -H "Authorization: Bearer your-api-key"
```

```python
import requests

agent_id = "agent_550e8400e29b41d4"
url = f"https://api.stateset.io/agents/{agent_id}"
headers = {"Authorization": "Bearer your-api-key"}

response = requests.get(url, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "agent_id": "agent_550e8400e29b41d4",
  "model_name": "gpt2",
  "created_at": "2024-01-15T10:30:00.000Z",
  "conversation_count": 42,
  "total_tokens_used": 15234,
  "config": {
    "model_name": "gpt2",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "system_prompt": "You are a helpful customer support assistant.",
    "use_chat_template": true
  },
  "status": "active"
}
```

### Delete Agent

Delete an agent and all associated conversations.

#### Request

```bash
curl -X DELETE https://api.stateset.io/agents/agent_550e8400e29b41d4 \
  -H "Authorization: Bearer your-api-key"
```

```python
import requests

agent_id = "agent_550e8400e29b41d4"
url = f"https://api.stateset.io/agents/{agent_id}"
headers = {"Authorization": "Bearer your-api-key"}

response = requests.delete(url, headers=headers)
print(response.status_code)  # 204
```

#### Response (204 No Content)

No response body. Status code 204 indicates successful deletion.

---

## Conversations

### Send Message (Single-turn)

Send a single message to an agent and get a response.

#### Request

```bash
curl -X POST https://api.stateset.io/conversations \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is the status of my order #12345?"
      }
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }'
```

```python
import requests

url = "https://api.stateset.io/conversations"
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

payload = {
    "messages": [
        {
            "role": "user",
            "content": "What is the status of my order #12345?"
        }
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": False
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "conversation_id": "conv_772e9488b40d53f6",
  "response": "I'd be happy to help you check the status of order #12345. Let me look that up for you. Your order was shipped on January 14th and is currently in transit. The expected delivery date is January 17th. You can track your package using tracking number TRK789456123.",
  "tokens_used": 52,
  "processing_time": 1.23,
  "metadata": {
    "model": "gpt2",
    "finish_reason": "stop"
  },
  "request_id": "req_def456uvw012",
  "timestamp": "2024-01-15T10:40:00.000Z"
}
```

### Multi-turn Conversation

Continue a conversation with context.

#### Request

```bash
curl -X POST https://api.stateset.io/conversations \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is machine learning?"
      },
      {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
      },
      {
        "role": "user",
        "content": "Can you give me an example?"
      }
    ],
    "conversation_id": "conv_772e9488b40d53f6",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

```python
import requests

url = "https://api.stateset.io/conversations"
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

payload = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is machine learning?"
        },
        {
            "role": "assistant",
            "content": "Machine learning is a subset of artificial intelligence..."
        },
        {
            "role": "user",
            "content": "Can you give me an example?"
        }
    ],
    "conversation_id": "conv_772e9488b40d53f6",
    "max_tokens": 256,
    "temperature": 0.7
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "conversation_id": "conv_772e9488b40d53f6",
  "response": "Certainly! A common example is email spam filtering. The system is trained on thousands of emails labeled as 'spam' or 'not spam'. Over time, it learns patterns that distinguish spam from legitimate emails - like certain keywords, sender patterns, or formatting. Once trained, it can automatically classify new emails without being told specific rules.",
  "tokens_used": 68,
  "processing_time": 1.45,
  "metadata": {
    "model": "gpt2",
    "finish_reason": "stop",
    "turn_count": 3
  },
  "request_id": "req_ghi789pqr345",
  "timestamp": "2024-01-15T10:42:00.000Z"
}
```

### List Conversations

Get a paginated list of conversations.

#### Request

```bash
curl -X GET "https://api.stateset.io/conversations?page=1&page_size=20&agent_id=agent_550e8400e29b41d4" \
  -H "Authorization: Bearer your-api-key"
```

```python
import requests

url = "https://api.stateset.io/conversations"
headers = {"Authorization": "Bearer your-api-key"}
params = {
    "page": 1,
    "page_size": 20,
    "agent_id": "agent_550e8400e29b41d4"
}

response = requests.get(url, headers=headers, params=params)
print(response.json())
```

#### Response (200 OK)

```json
{
  "items": [
    {
      "conversation_id": "conv_772e9488b40d53f6",
      "agent_id": "agent_550e8400e29b41d4",
      "message_count": 6,
      "created_at": "2024-01-15T10:40:00.000Z",
      "last_message_at": "2024-01-15T10:45:00.000Z",
      "total_tokens": 324
    },
    {
      "conversation_id": "conv_883f0599c51e64g7",
      "agent_id": "agent_550e8400e29b41d4",
      "message_count": 4,
      "created_at": "2024-01-15T09:20:00.000Z",
      "last_message_at": "2024-01-15T09:28:00.000Z",
      "total_tokens": 198
    }
  ],
  "total": 42,
  "page": 1,
  "page_size": 20,
  "has_next": true,
  "has_prev": false,
  "request_id": "req_jkl012stu678",
  "timestamp": "2024-01-15T10:50:00.000Z"
}
```

### Get Conversation Details

Get full details and message history of a conversation.

#### Request

```bash
curl -X GET https://api.stateset.io/conversations/conv_772e9488b40d53f6 \
  -H "Authorization: Bearer your-api-key"
```

```python
import requests

conversation_id = "conv_772e9488b40d53f6"
url = f"https://api.stateset.io/conversations/{conversation_id}"
headers = {"Authorization": "Bearer your-api-key"}

response = requests.get(url, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "conversation_id": "conv_772e9488b40d53f6",
  "agent_id": "agent_550e8400e29b41d4",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant.",
      "timestamp": "2024-01-15T10:40:00.000Z"
    },
    {
      "role": "user",
      "content": "What is machine learning?",
      "timestamp": "2024-01-15T10:40:05.000Z"
    },
    {
      "role": "assistant",
      "content": "Machine learning is a subset of artificial intelligence...",
      "timestamp": "2024-01-15T10:40:07.000Z"
    },
    {
      "role": "user",
      "content": "Can you give me an example?",
      "timestamp": "2024-01-15T10:42:00.000Z"
    },
    {
      "role": "assistant",
      "content": "Certainly! A common example is email spam filtering...",
      "timestamp": "2024-01-15T10:42:02.000Z"
    }
  ],
  "created_at": "2024-01-15T10:40:00.000Z",
  "last_message_at": "2024-01-15T10:42:02.000Z",
  "total_tokens": 324,
  "metadata": {
    "model": "gpt2",
    "temperature": 0.7
  }
}
```

### Delete Conversation

Delete a specific conversation.

#### Request

```bash
curl -X DELETE https://api.stateset.io/conversations/conv_772e9488b40d53f6 \
  -H "Authorization: Bearer your-api-key"
```

```python
import requests

conversation_id = "conv_772e9488b40d53f6"
url = f"https://api.stateset.io/conversations/{conversation_id}"
headers = {"Authorization": "Bearer your-api-key"}

response = requests.delete(url, headers=headers)
print(response.status_code)  # 204
```

#### Response (204 No Content)

No response body. Status code 204 indicates successful deletion.

---

## Training

### Start Training Job

Start a new reinforcement learning training job.

#### Request

```bash
curl -X POST https://api.stateset.io/training \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_config": {
      "model_name": "gpt2",
      "max_new_tokens": 256,
      "temperature": 0.7
    },
    "environment_scenarios": [
      {
        "id": "customer_support_1",
        "topic": "order_status",
        "user_responses": [
          "What is the status of my order?",
          "When will it arrive?",
          "Thank you!"
        ]
      },
      {
        "id": "customer_support_2",
        "topic": "returns",
        "user_responses": [
          "I want to return my item",
          "How long does a refund take?",
          "Okay, thanks"
        ]
      }
    ],
    "reward_config": {
      "helpfulness_weight": 0.7,
      "safety_weight": 0.2,
      "efficiency_weight": 0.1
    },
    "num_episodes": 100,
    "profile": "balanced"
  }'
```

```python
import requests

url = "https://api.stateset.io/training"
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

payload = {
    "agent_config": {
        "model_name": "gpt2",
        "max_new_tokens": 256,
        "temperature": 0.7
    },
    "environment_scenarios": [
        {
            "id": "customer_support_1",
            "topic": "order_status",
            "user_responses": [
                "What is the status of my order?",
                "When will it arrive?",
                "Thank you!"
            ]
        },
        {
            "id": "customer_support_2",
            "topic": "returns",
            "user_responses": [
                "I want to return my item",
                "How long does a refund take?",
                "Okay, thanks"
            ]
        }
    ],
    "reward_config": {
        "helpfulness_weight": 0.7,
        "safety_weight": 0.2,
        "efficiency_weight": 0.1
    },
    "num_episodes": 100,
    "profile": "balanced"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

#### Response (202 Accepted)

```json
{
  "training_id": "train_994g1600d62f75h8",
  "status": "running",
  "message": "Training started successfully",
  "estimated_time": 3000,
  "request_id": "req_mno345vwx901",
  "timestamp": "2024-01-15T11:00:00.000Z"
}
```

### Get Training Status

Monitor the progress of a training job.

#### Request

```bash
curl -X GET https://api.stateset.io/training/train_994g1600d62f75h8 \
  -H "Authorization: Bearer your-api-key"
```

```python
import requests

training_id = "train_994g1600d62f75h8"
url = f"https://api.stateset.io/training/{training_id}"
headers = {"Authorization": "Bearer your-api-key"}

response = requests.get(url, headers=headers)
print(response.json())
```

#### Response (200 OK) - Training In Progress

```json
{
  "training_id": "train_994g1600d62f75h8",
  "status": "running",
  "created_at": "2024-01-15T11:00:00.000Z",
  "started_at": "2024-01-15T11:00:05.000Z",
  "completed_at": null,
  "progress": 45.5,
  "current_episode": 45,
  "total_episodes": 100,
  "metrics": {
    "average_reward": 0.73,
    "loss": 0.342,
    "learning_rate": 0.0001,
    "episodes_completed": 45,
    "total_trajectories": 450,
    "computation_time_seconds": 1350.5
  },
  "config": {
    "model_name": "gpt2",
    "num_episodes": 100,
    "profile": "balanced"
  }
}
```

#### Response (200 OK) - Training Complete

```json
{
  "training_id": "train_994g1600d62f75h8",
  "status": "completed",
  "created_at": "2024-01-15T11:00:00.000Z",
  "started_at": "2024-01-15T11:00:05.000Z",
  "completed_at": "2024-01-15T11:52:30.000Z",
  "progress": 100.0,
  "current_episode": 100,
  "total_episodes": 100,
  "metrics": {
    "average_reward": 0.85,
    "final_loss": 0.156,
    "learning_rate": 0.00005,
    "episodes_completed": 100,
    "total_trajectories": 1000,
    "computation_time_seconds": 3145.2,
    "reward_progression": [0.45, 0.62, 0.71, 0.78, 0.83, 0.85],
    "model_checkpoint": "s3://models/train_994g1600d62f75h8/final"
  },
  "config": {
    "model_name": "gpt2",
    "num_episodes": 100,
    "profile": "balanced"
  }
}
```

### List Training Jobs

Get a paginated list of training jobs.

#### Request

```bash
curl -X GET "https://api.stateset.io/training?page=1&page_size=20&status=running" \
  -H "Authorization: Bearer your-api-key"
```

```python
import requests

url = "https://api.stateset.io/training"
headers = {"Authorization": "Bearer your-api-key"}
params = {
    "page": 1,
    "page_size": 20,
    "status": "running"
}

response = requests.get(url, headers=headers, params=params)
print(response.json())
```

#### Response (200 OK)

```json
{
  "items": [
    {
      "training_id": "train_994g1600d62f75h8",
      "status": "running",
      "created_at": "2024-01-15T11:00:00.000Z",
      "started_at": "2024-01-15T11:00:05.000Z",
      "completed_at": null,
      "progress": 45.5,
      "current_episode": 45,
      "total_episodes": 100,
      "metrics": {
        "average_reward": 0.73
      },
      "config": {
        "model_name": "gpt2",
        "profile": "balanced"
      }
    },
    {
      "training_id": "train_aa5h2711e73g86i9",
      "status": "running",
      "created_at": "2024-01-15T10:45:00.000Z",
      "started_at": "2024-01-15T10:45:10.000Z",
      "completed_at": null,
      "progress": 78.0,
      "current_episode": 78,
      "total_episodes": 100,
      "metrics": {
        "average_reward": 0.81
      },
      "config": {
        "model_name": "gpt2",
        "profile": "performance"
      }
    }
  ],
  "total": 2,
  "page": 1,
  "page_size": 20,
  "has_next": false,
  "has_prev": false,
  "request_id": "req_pqr678zab234",
  "timestamp": "2024-01-15T11:30:00.000Z"
}
```

### Cancel Training Job

Stop a running training job.

#### Request

```bash
curl -X DELETE https://api.stateset.io/training/train_994g1600d62f75h8 \
  -H "Authorization: Bearer your-api-key"
```

```python
import requests

training_id = "train_994g1600d62f75h8"
url = f"https://api.stateset.io/training/{training_id}"
headers = {"Authorization": "Bearer your-api-key"}

response = requests.delete(url, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "training_id": "train_994g1600d62f75h8",
  "status": "cancelled",
  "message": "Training job cancelled successfully"
}
```

---

## Health & Monitoring

### Health Check

Check the overall health of the API service.

#### Request

```bash
curl -X GET https://api.stateset.io/health
```

```python
import requests

url = "https://api.stateset.io/health"
response = requests.get(url)
print(response.json())
```

#### Response (200 OK)

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2024-01-15T12:00:00.000Z",
  "checks": {
    "api": {
      "status": "healthy",
      "latency_ms": 2.5,
      "message": "API responding normally"
    },
    "agent_service": {
      "status": "healthy",
      "latency_ms": 5.2,
      "message": null
    },
    "training_service": {
      "status": "healthy",
      "latency_ms": 3.8,
      "message": null
    },
    "security_monitor": {
      "status": "healthy",
      "latency_ms": 1.1,
      "message": null
    }
  }
}
```

### Ready Probe (Kubernetes)

Check if the service is ready to accept traffic.

#### Request

```bash
curl -X GET https://api.stateset.io/ready
```

```python
import requests

url = "https://api.stateset.io/ready"
response = requests.get(url)
print(response.json())
```

#### Response (200 OK)

```json
{
  "status": "ready"
}
```

### Liveness Probe (Kubernetes)

Check if the service is alive.

#### Request

```bash
curl -X GET https://api.stateset.io/live
```

```python
import requests

url = "https://api.stateset.io/live"
response = requests.get(url)
print(response.json())
```

#### Response (200 OK)

```json
{
  "status": "alive",
  "timestamp": "2024-01-15T12:05:00.000Z"
}
```

### Get Metrics

Get comprehensive system metrics (requires admin role).

#### Request

```bash
curl -X GET https://api.stateset.io/metrics \
  -H "Authorization: Bearer your-admin-api-key"
```

```python
import requests

url = "https://api.stateset.io/metrics"
headers = {"Authorization": "Bearer your-admin-api-key"}

response = requests.get(url, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "timestamp": "2024-01-15T12:10:00.000Z",
  "system_metrics": {
    "total_operations": 15234,
    "active_operations": 23,
    "uptime_seconds": 864532.5
  },
  "api_metrics": {
    "total_requests": 45892,
    "requests_by_endpoint": {
      "/agents": 12340,
      "/conversations": 28450,
      "/training": 4102,
      "/metrics": 1000
    },
    "status_codes": {
      "200": 42156,
      "201": 2456,
      "204": 890,
      "400": 234,
      "401": 45,
      "404": 111
    },
    "errors_by_endpoint": {
      "/agents": 12,
      "/conversations": 45,
      "/training": 23
    },
    "rate_limit_hits": 156
  },
  "performance_metrics": {
    "latency": {
      "p50": 125.5,
      "p95": 450.2,
      "p99": 890.7,
      "avg": 178.3
    },
    "operations_by_type": {
      "agent_creation": 1234,
      "conversation": 8920,
      "training": 456
    }
  },
  "security_metrics": {
    "total_events": 892,
    "events_last_hour": 45,
    "blocked_events": 23,
    "active_lockouts": 2,
    "high_threat_events": 5,
    "events_by_type": {
      "rate_limit_exceeded": 156,
      "invalid_auth": 45,
      "prompt_injection_attempt": 12,
      "suspicious_pattern": 8
    }
  },
  "cache_metrics": {
    "hits": 34567,
    "misses": 4321,
    "hit_rate": 0.889,
    "size": 2048576,
    "evictions": 234
  },
  "request_id": "req_stu901cde567",
  "timestamp": "2024-01-15T12:10:00.000Z"
}
```

### Get Security Metrics

Get detailed security-specific metrics (requires admin role).

#### Request

```bash
curl -X GET https://api.stateset.io/metrics/security \
  -H "Authorization: Bearer your-admin-api-key"
```

```python
import requests

url = "https://api.stateset.io/metrics/security"
headers = {"Authorization": "Bearer your-admin-api-key"}

response = requests.get(url, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "timestamp": "2024-01-15T12:15:00.000Z",
  "statistics": {
    "total_events": 892,
    "events_last_hour": 45,
    "blocked_events": 23,
    "active_lockouts": 2,
    "high_threat_events": 5,
    "events_by_type": {
      "rate_limit_exceeded": 156,
      "invalid_auth": 45,
      "prompt_injection_attempt": 12,
      "suspicious_pattern": 8,
      "blocked_request": 23
    }
  },
  "recent_events": [
    {
      "type": "prompt_injection_attempt",
      "threat_level": "high",
      "blocked": true,
      "path": "/conversations",
      "timestamp": "2024-01-15T12:14:30.000Z"
    },
    {
      "type": "rate_limit_exceeded",
      "threat_level": "medium",
      "blocked": true,
      "path": "/agents",
      "timestamp": "2024-01-15T12:14:15.000Z"
    },
    {
      "type": "invalid_auth",
      "threat_level": "medium",
      "blocked": true,
      "path": "/training",
      "timestamp": "2024-01-15T12:13:45.000Z"
    }
  ]
}
```

### Get Cache Metrics

Get cache statistics (requires admin role).

#### Request

```bash
curl -X GET https://api.stateset.io/metrics/cache \
  -H "Authorization: Bearer your-admin-api-key"
```

```python
import requests

url = "https://api.stateset.io/metrics/cache"
headers = {"Authorization": "Bearer your-admin-api-key"}

response = requests.get(url, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "timestamp": "2024-01-15T12:20:00.000Z",
  "cache": {
    "hits": 34567,
    "misses": 4321,
    "hit_rate": 0.889,
    "size": 2048576,
    "max_size": 10485760,
    "evictions": 234,
    "entries": 1456,
    "memory_usage_bytes": 2048576
  }
}
```

### Circuit Breaker Status

Get the status of all circuit breakers.

#### Request

```bash
curl -X GET https://api.stateset.io/circuits
```

```python
import requests

url = "https://api.stateset.io/circuits"
response = requests.get(url)
print(response.json())
```

#### Response (200 OK)

```json
{
  "circuits": {
    "agent_service": {
      "state": "closed",
      "failure_count": 2,
      "success_count": 1245,
      "last_failure_time": "2024-01-15T10:30:00.000Z",
      "next_attempt_time": null
    },
    "training_service": {
      "state": "closed",
      "failure_count": 0,
      "success_count": 456,
      "last_failure_time": null,
      "next_attempt_time": null
    },
    "external_api": {
      "state": "half_open",
      "failure_count": 3,
      "success_count": 892,
      "last_failure_time": "2024-01-15T12:15:00.000Z",
      "next_attempt_time": "2024-01-15T12:20:00.000Z"
    }
  },
  "timestamp": "2024-01-15T12:18:00.000Z"
}
```

---

## Batch Operations

### Batch Training

Submit multiple training jobs in a single request.

#### Request

```bash
curl -X POST https://api.stateset.io/v1/batch/train \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "prompts": ["What is artificial intelligence?"],
        "strategy": "computational",
        "num_iterations": 10,
        "idempotency_key": "batch_item_1"
      },
      {
        "prompts": ["Explain machine learning"],
        "strategy": "computational",
        "num_iterations": 10,
        "idempotency_key": "batch_item_2"
      },
      {
        "prompts": ["What are neural networks?"],
        "strategy": "computational",
        "num_iterations": 10,
        "idempotency_key": "batch_item_3"
      }
    ],
    "parallel": true,
    "max_concurrent": 10,
    "fail_fast": false
  }'
```

```python
import requests

url = "https://api.stateset.io/v1/batch/train"
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

payload = {
    "items": [
        {
            "prompts": ["What is artificial intelligence?"],
            "strategy": "computational",
            "num_iterations": 10,
            "idempotency_key": "batch_item_1"
        },
        {
            "prompts": ["Explain machine learning"],
            "strategy": "computational",
            "num_iterations": 10,
            "idempotency_key": "batch_item_2"
        },
        {
            "prompts": ["What are neural networks?"],
            "strategy": "computational",
            "num_iterations": 10,
            "idempotency_key": "batch_item_3"
        }
    ],
    "parallel": True,
    "max_concurrent": 10,
    "fail_fast": False
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "batch_id": "batch_bb6i3822f84h97j0",
  "total_items": 3,
  "accepted": 3,
  "rejected": 0,
  "results": [
    {
      "index": 0,
      "job_id": "train_cc7j4933g95i08k1",
      "status": "accepted",
      "error": null
    },
    {
      "index": 1,
      "job_id": "train_dd8k5044h06j19l2",
      "status": "accepted",
      "error": null
    },
    {
      "index": 2,
      "job_id": "train_ee9l6155i17k20m3",
      "status": "accepted",
      "error": null
    }
  ]
}
```

### Batch Job Status

Check the status of multiple training jobs.

#### Request

```bash
curl -X POST https://api.stateset.io/v1/batch/status \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "job_ids": [
      "train_cc7j4933g95i08k1",
      "train_dd8k5044h06j19l2",
      "train_ee9l6155i17k20m3"
    ]
  }'
```

```python
import requests

url = "https://api.stateset.io/v1/batch/status"
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

payload = {
    "job_ids": [
        "train_cc7j4933g95i08k1",
        "train_dd8k5044h06j19l2",
        "train_ee9l6155i17k20m3"
    ]
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "jobs": {
    "train_cc7j4933g95i08k1": {
      "job_id": "train_cc7j4933g95i08k1",
      "status": "completed",
      "iterations_completed": 10,
      "total_trajectories": 100,
      "average_reward": 0.82,
      "computation_used": 85.5,
      "started_at": "2024-01-15T12:30:00.000Z",
      "completed_at": "2024-01-15T12:35:30.000Z"
    },
    "train_dd8k5044h06j19l2": {
      "job_id": "train_dd8k5044h06j19l2",
      "status": "running",
      "iterations_completed": 7,
      "total_trajectories": 70,
      "average_reward": 0.75,
      "computation_used": 62.3,
      "started_at": "2024-01-15T12:30:00.000Z",
      "completed_at": null
    },
    "train_ee9l6155i17k20m3": {
      "job_id": "train_ee9l6155i17k20m3",
      "status": "running",
      "iterations_completed": 6,
      "total_trajectories": 60,
      "average_reward": 0.71,
      "computation_used": 55.8,
      "started_at": "2024-01-15T12:30:00.000Z",
      "completed_at": null
    }
  },
  "not_found": []
}
```

### Batch Cancel

Cancel multiple training jobs.

#### Request

```bash
curl -X POST https://api.stateset.io/v1/batch/cancel \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "job_ids": [
      "train_dd8k5044h06j19l2",
      "train_ee9l6155i17k20m3"
    ]
  }'
```

```python
import requests

url = "https://api.stateset.io/v1/batch/cancel"
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

payload = {
    "job_ids": [
        "train_dd8k5044h06j19l2",
        "train_ee9l6155i17k20m3"
    ]
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

#### Response (200 OK)

```json
{
  "cancelled": [
    "train_dd8k5044h06j19l2",
    "train_ee9l6155i17k20m3"
  ],
  "not_found": [],
  "already_completed": []
}
```

---

## Error Handling

The API uses standard HTTP status codes and returns detailed error information.

### Error Response Format

All errors follow this format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "field": "additional context",
    "reason": "detailed explanation"
  },
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

### Common Error Responses

#### 400 Bad Request - Invalid Input

```json
{
  "error": "ValidationError",
  "message": "Invalid request parameters",
  "details": {
    "field": "temperature",
    "reason": "temperature must be between 0.0 and 2.0"
  },
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

#### 401 Unauthorized - Missing/Invalid Authentication

```json
{
  "error": "UnauthorizedError",
  "message": "Authentication required. Provide an API key or bearer token.",
  "details": null,
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

#### 403 Forbidden - Insufficient Permissions

```json
{
  "error": "ForbiddenError",
  "message": "Insufficient permissions. Required role: admin",
  "details": {
    "user_roles": ["user"],
    "required_roles": ["admin"]
  },
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

#### 404 Not Found - Resource Not Found

```json
{
  "error": "NotFoundError",
  "message": "Agent not found: agent_invalid123",
  "details": {
    "resource_type": "agent",
    "resource_id": "agent_invalid123"
  },
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

#### 409 Conflict - Idempotency Key Conflict

```json
{
  "error": "ConflictError",
  "message": "Request with this idempotency key already exists",
  "details": {
    "idempotency_key": "unique_key_123",
    "existing_job_id": "train_ff0m7266j28l31n4"
  },
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

#### 429 Too Many Requests - Rate Limit Exceeded

```json
{
  "error": "RateLimitError",
  "message": "Rate limit exceeded. Try again in 60 seconds.",
  "details": {
    "limit": 100,
    "window_seconds": 60,
    "retry_after": 60
  },
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

#### 500 Internal Server Error

```json
{
  "error": "InternalError",
  "message": "An unexpected error occurred. Please try again later.",
  "details": {
    "incident_id": "inc_gg1n8377k39m42o5"
  },
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

#### 503 Service Unavailable

```json
{
  "error": "ServiceUnavailableError",
  "message": "Service temporarily unavailable. Please try again later.",
  "details": {
    "reason": "Scheduled maintenance",
    "retry_after": 300
  },
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

### Security Error - Prompt Injection Detected

```json
{
  "error": "PromptInjectionError",
  "message": "Potential prompt injection detected in messages[0].content",
  "details": {
    "field": "messages[0].content",
    "matched_patterns": [
      "ignore previous instructions",
      "system override"
    ]
  },
  "timestamp": "2024-01-15T12:00:00.000Z",
  "request_id": "req_xyz789abc123"
}
```

### Error Handling Best Practices

1. **Always check status codes** - Don't rely solely on parsing the response body
2. **Use request_id for support** - Include the request_id when contacting support
3. **Implement exponential backoff** - For 429 and 503 errors
4. **Handle idempotency** - Save idempotency keys to retry safely
5. **Parse error details** - Use the details field for specific error handling

#### Python Error Handling Example

```python
import requests
import time

def make_api_request(url, headers, payload, max_retries=3):
    """Make API request with error handling and retries."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)

            # Success
            if response.status_code in [200, 201, 202]:
                return response.json()

            # Rate limited - wait and retry
            elif response.status_code == 429:
                error_data = response.json()
                retry_after = error_data.get("details", {}).get("retry_after", 60)
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue

            # Service unavailable - exponential backoff
            elif response.status_code == 503:
                wait_time = 2 ** attempt
                print(f"Service unavailable. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            # Client errors - don't retry
            elif response.status_code in [400, 401, 403, 404]:
                error_data = response.json()
                raise ValueError(f"API error: {error_data['message']}")

            # Other errors
            else:
                error_data = response.json()
                print(f"Error: {error_data}")
                raise RuntimeError(f"API request failed: {error_data['message']}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError("Max retries exceeded")

# Usage
try:
    result = make_api_request(
        url="https://api.stateset.io/training",
        headers={"Authorization": "Bearer your-api-key"},
        payload={"prompts": ["Test"], "strategy": "computational"}
    )
    print("Success:", result)
except Exception as e:
    print("Failed:", e)
```

---

## Additional Resources

- [API Reference Documentation](./API_REFERENCE.md)
- [Comprehensive Usage Guide](./COMPREHENSIVE_USAGE_GUIDE.md)
- [Architecture Overview](./ARCHITECTURE.md)
- [OpenAPI Specification](https://api.stateset.io/openapi.json)
- [Interactive API Docs](https://api.stateset.io/docs)

## Support

For API support, please:
1. Check the [documentation](https://docs.stateset.io)
2. Include your `request_id` from error responses
3. Contact support@stateset.io

## Rate Limits

Default rate limits (vary by plan):
- **Free tier**: 100 requests per minute
- **Pro tier**: 1,000 requests per minute
- **Enterprise**: Custom limits

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets
