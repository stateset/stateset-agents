# GRPO Agent Framework - CLI Reference

The GRPO Agent Framework provides command-line tools for training, evaluating, and serving agents.

## Installation

Once installed, the following CLI commands are available:

- `grpo-train`: Train an agent
- `grpo-evaluate`: Evaluate a trained agent
- `grpo-serve`: Serve a trained agent via API

## Commands

### grpo-train

Train a multi-turn conversational agent using GRPO.

```bash
grpo-train [OPTIONS] CONFIG_FILE
```

#### Arguments

- `CONFIG_FILE`: Path to training configuration file (YAML or JSON)

#### Options

```bash
--model-name TEXT          Model name or path [default: openai/gpt-oss-120b]
--output-dir TEXT          Output directory for checkpoints [default: ./outputs]
--num-episodes INTEGER     Number of training episodes [default: 1000]
--profile TEXT             Training profile: conservative, balanced, aggressive [default: balanced]
--auto-adjust              Enable automatic hyperparameter adjustment
--early-stopping           Enable early stopping
--patience INTEGER         Early stopping patience [default: 50]
--save-every INTEGER       Save checkpoint every N episodes [default: 100]
--eval-every INTEGER       Run evaluation every N episodes [default: 50]
--log-level TEXT           Logging level [default: INFO]
--wandb-project TEXT       Weights & Biases project name
--no-gpu                   Force CPU training
--help                     Show help message
```

#### Examples

```bash
# Basic training
grpo-train configs/customer_service.yaml

# Training with custom parameters
grpo-train configs/tutoring.yaml \
  --model-name openai/gpt-oss-120b \
  --num-episodes 2000 \
  --profile aggressive \
  --auto-adjust

# Training with W&B logging
grpo-train configs/my_agent.yaml \
  --wandb-project my-grpo-experiments \
  --output-dir ./checkpoints/experiment_1
```

#### Configuration File Format

```yaml
# Example: customer_service.yaml
agent:
  type: "multi_turn"
  model_name: "openai/gpt-oss-120b"
  system_prompt: "You are a helpful customer service representative."
  temperature: 0.7
  max_new_tokens: 256
  memory_window: 8

environment:
  type: "conversation"
  max_turns: 12
  scenarios_file: "data/customer_service_scenarios.json"
  persona: "You are a customer with a specific need or problem."

rewards:
  - type: "helpfulness"
    weight: 0.4
  - type: "safety" 
    weight: 0.3
  - type: "conciseness"
    weight: 0.2
  - type: "engagement"
    weight: 0.1

training:
  profile: "balanced"
  num_episodes: 1000
  auto_adjust: true
  early_stopping: true
  patience: 50
  
  # Optional: Override specific parameters
  overrides:
    learning_rate: 5e-6
    num_generations: 16
    max_grad_norm: 1.0
```

### grpo-evaluate

Evaluate a trained agent's performance.

```bash
grpo-evaluate [OPTIONS] CHECKPOINT_PATH
```

#### Arguments

- `CHECKPOINT_PATH`: Path to saved agent checkpoint

#### Options

```bash
--config-file TEXT         Evaluation configuration file
--num-episodes INTEGER     Number of evaluation episodes [default: 100]
--scenarios-file TEXT      Custom scenarios file for evaluation
--output-file TEXT         Save results to file
--metrics TEXT             Comma-separated metrics to compute [default: reward,length,success_rate]
--verbose                  Show detailed episode results
--compare-with TEXT        Compare with another checkpoint
--help                     Show help message
```

#### Examples

```bash
# Basic evaluation
grpo-evaluate ./checkpoints/customer_service_agent

# Detailed evaluation with custom scenarios
grpo-evaluate ./checkpoints/my_agent \
  --scenarios-file data/test_scenarios.json \
  --num-episodes 200 \
  --verbose \
  --output-file results.json

# Compare two models
grpo-evaluate ./checkpoints/model_v2 \
  --compare-with ./checkpoints/model_v1 \
  --metrics reward,length,success_rate,engagement
```

#### Output Example

```
Agent Evaluation Results
========================
Checkpoint: ./checkpoints/customer_service_agent
Episodes: 100
Evaluation Time: 15.3 minutes

Performance Metrics:
- Average Reward: 0.847 ± 0.123
- Success Rate: 89.2%
- Average Episode Length: 8.3 ± 2.1 turns
- Engagement Score: 0.762 ± 0.089

Detailed Breakdown:
- Helpfulness: 0.891 ± 0.076
- Safety: 0.967 ± 0.043
- Conciseness: 0.734 ± 0.156
- Task Completion: 0.892 ± 0.089

Episode Length Distribution:
- 1-5 turns: 23%
- 6-10 turns: 58%
- 11-15 turns: 19%

Top Performing Scenarios:
1. Product Inquiry: 0.934 avg reward
2. Technical Support: 0.876 avg reward
3. Billing Questions: 0.823 avg reward

Improvement Opportunities:
- Return Requests: 0.721 avg reward (below average)
- Complex Technical Issues: 0.698 avg reward
```

### grpo-serve

Serve a trained agent via REST API.

```bash
grpo-serve [OPTIONS] CHECKPOINT_PATH
```

#### Arguments

- `CHECKPOINT_PATH`: Path to saved agent checkpoint

#### Options

```bash
--host TEXT                Host address [default: localhost]
--port INTEGER             Port number [default: 8000]
--max-conversations INTEGER   Maximum concurrent conversations [default: 100]
--conversation-timeout INTEGER   Conversation timeout in minutes [default: 30]
--log-conversations        Log conversations to file
--auth-token TEXT          Require authentication token
--cors-origins TEXT        Comma-separated CORS origins
--help                     Show help message
```

#### Examples

```bash
# Basic serving
grpo-serve ./checkpoints/customer_service_agent

# Production serving with authentication
grpo-serve ./checkpoints/my_agent \
  --host 0.0.0.0 \
  --port 8080 \
  --auth-token my-secret-token \
  --log-conversations \
  --cors-origins "https://myapp.com,https://admin.myapp.com"

# Development serving
grpo-serve ./checkpoints/dev_agent \
  --port 3000 \
  --max-conversations 10 \
  --conversation-timeout 5
```

#### API Endpoints

Once running, the server provides the following endpoints:

##### POST /chat/start
Start a new conversation.

```bash
curl -X POST http://localhost:8000/chat/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "context": {"topic": "support"}}'
```

Response:
```json
{
  "conversation_id": "conv_abc123",
  "status": "started",
  "message": "Hello! How can I help you today?"
}
```

##### POST /chat/{conversation_id}/message
Send a message in an existing conversation.

```bash
curl -X POST http://localhost:8000/chat/conv_abc123/message \
  -H "Content-Type: application/json" \
  -d '{"message": "I need help with my order"}'
```

Response:
```json
{
  "conversation_id": "conv_abc123",
  "response": "I'd be happy to help you with your order. Could you please provide your order number?",
  "turn_count": 2,
  "status": "ongoing"
}
```

##### GET /chat/{conversation_id}/history
Get conversation history.

```bash
curl http://localhost:8000/chat/conv_abc123/history
```

Response:
```json
{
  "conversation_id": "conv_abc123",
  "turns": [
    {"role": "assistant", "content": "Hello! How can I help you today?", "timestamp": "2024-01-15T10:30:00Z"},
    {"role": "user", "content": "I need help with my order", "timestamp": "2024-01-15T10:30:15Z"},
    {"role": "assistant", "content": "I'd be happy to help...", "timestamp": "2024-01-15T10:30:18Z"}
  ],
  "status": "ongoing"
}
```

##### DELETE /chat/{conversation_id}
End a conversation.

```bash
curl -X DELETE http://localhost:8000/chat/conv_abc123
```

##### GET /health
Health check endpoint.

```bash
curl http://localhost:8000/health
```

##### GET /metrics
Get server metrics.

```bash
curl http://localhost:8000/metrics
```

## Configuration Files

### Training Configuration

Detailed training configuration options:

```yaml
# Complete training configuration example
agent:
  type: "multi_turn"  # or "tool", "custom"
  model_name: "openai/gpt-oss-120b"
  
  # Agent behavior
  system_prompt: "You are a helpful AI assistant."
  temperature: 0.8
  max_new_tokens: 512
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  
  # Multi-turn specific
  memory_window: 10
  context_compression: false
  
  # Tool agent specific (if type: "tool")
  tools:
    - name: "calculator"
      description: "Perform calculations"
      function: "calculator_function"

environment:
  type: "conversation"  # or "task", "custom"
  max_turns: 15
  
  # For conversation environment
  scenarios_file: "data/scenarios.json"
  persona: "You are a helpful user."
  
  # For task environment
  tasks_file: "data/tasks.json"
  success_criteria: "task_completion_function"

rewards:
  # Built-in rewards
  - type: "helpfulness"
    weight: 0.4
  - type: "safety"
    weight: 0.3
  - type: "correctness"
    weight: 0.2
    ground_truth_file: "data/ground_truth.json"
  - type: "conciseness"
    weight: 0.1
    optimal_length: 150
  
  # Custom reward
  - type: "custom"
    module: "my_rewards.politeness_reward"
    weight: 0.2
    
  # Composite reward settings
  combination_method: "weighted_sum"  # or "average", "min", "max"

training:
  # Profile and episodes
  profile: "balanced"  # or "conservative", "aggressive", "experimental"
  num_episodes: 1000
  
  # Optimization
  auto_adjust: true
  early_stopping: true
  patience: 50
  
  # Checkpointing
  save_every: 100
  eval_every: 50
  
  # Advanced options
  multi_gpu: false
  gradient_checkpointing: false
  mixed_precision: true
  
  # Custom overrides
  overrides:
    learning_rate: 5e-6
    num_generations: 16
    batch_size: 4
    gradient_accumulation_steps: 4
    max_grad_norm: 1.0
    warmup_steps: 100
    weight_decay: 0.01

logging:
  level: "INFO"
  log_file: "training.log"
  
  # Weights & Biases
  wandb:
    project: "my-grpo-project"
    entity: "my-team"
    tags: ["experiment", "customer-service"]
    
  # TensorBoard
  tensorboard:
    log_dir: "./logs"

# Environment variables
environment_variables:
  CUDA_VISIBLE_DEVICES: "0,1"
  TOKENIZERS_PARALLELISM: "false"
```

### Scenarios File Format

```json
{
  "scenarios": [
    {
      "id": "customer_inquiry_1",
      "type": "product_question",
      "context": "Customer interested in product features",
      "metadata": {
        "difficulty": "easy",
        "category": "sales",
        "expected_turns": 5
      },
      "initial_state": {
        "user_goal": "Learn about product X",
        "user_knowledge": "beginner"
      },
      "user_responses": [
        "Hi, I'm interested in your new product.",
        "What makes it different from competitors?",
        "That sounds interesting. What's the price?",
        "Are there any current promotions?",
        "Great, how do I place an order?"
      ],
      "success_criteria": {
        "information_provided": ["features", "pricing", "ordering"],
        "user_satisfaction": "high"
      }
    }
  ]
}
```

## Environment Variables

Key environment variables for the CLI tools:

```bash
# Model and training
export GRPO_DEFAULT_MODEL="openai/gpt-oss-120b"
export GRPO_CACHE_DIR="~/.cache/grpo"
export GRPO_DATA_DIR="./data"

# Hardware
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export GRPO_USE_GPU="true"
export GRPO_MIXED_PRECISION="true"

# Logging
export GRPO_LOG_LEVEL="INFO"
export WANDB_PROJECT="my-grpo-experiments"
export WANDB_API_KEY="your-wandb-key"

# Serving
export GRPO_SERVER_HOST="0.0.0.0"
export GRPO_SERVER_PORT="8000"
export GRPO_AUTH_TOKEN="your-secret-token"
```

## Troubleshooting

### Common CLI Issues

1. **ImportError: No module named 'stateset_agents'**
   ```bash
   pip install grpo-agent-framework
   ```

2. **CUDA out of memory**
   ```bash
   # Use smaller batch size or gradient accumulation
   grpo-train config.yaml --profile conservative
   
   # Or force CPU training
   grpo-train config.yaml --no-gpu
   ```

3. **Permission denied when serving**
   ```bash
   # Use different port
   grpo-serve checkpoint --port 8080
   
   # Or run with sudo (not recommended)
   sudo grpo-serve checkpoint --port 80
   ```

4. **Configuration file not found**
   ```bash
   # Use absolute path
   grpo-train /full/path/to/config.yaml
   
   # Or check current directory
   ls -la *.yaml
   ```

### Getting Help

For each command, use `--help` to see all available options:

```bash
grpo-train --help
grpo-evaluate --help
grpo-serve --help
```

For more detailed documentation, visit: https://grpo-framework.readthedocs.io