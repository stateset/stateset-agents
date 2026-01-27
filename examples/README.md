# StateSet Agents Examples

Welcome to the StateSet Agents examples directory! This collection demonstrates how to use the framework for training conversational AI agents with Group Relative Policy Optimization (GRPO).

## 📚 Documentation

- **[Advanced Training Examples](./ADVANCED_TRAINING_README.md)** ⭐ NEW!
  - Distributed Multi-GPU Training
  - Custom Reward Functions
  - Advanced Optimization Techniques

- **[API Examples](./API_EXAMPLES_README.md)**
  - API Client Usage
  - Interactive Chatbot
  - WebSocket Integration

## 🚀 Quick Start Examples

### Hello World

The fastest way to get started - runs instantly with no downloads:

```bash
python examples/hello_world.py
```

**Features:**
- No model downloads required (uses stub mode)
- Complete agent creation and conversation flow
- Reward computation demonstration
- Training loop overview

### Quick Start

A simple example showing basic agent usage:

```bash
python examples/quick_start.py
```

**Features:**
- Real model usage (GPT-2)
- Basic conversation handling
- Environment setup
- Reward computation

## 🎓 Training Examples

### Basic Training

#### 1. Complete GRPO Training

Full-featured GRPO training example with all components:

```bash
python examples/complete_grpo_training.py
```

**Covers:**
- Agent initialization
- Environment creation
- Reward function setup
- Complete training loop
- Checkpoint saving

#### 2. TRL Integration

Using Hugging Face TRL library for GRPO:

```bash
python examples/train_with_trl_grpo.py
```

**Features:**
- TRL GRPO trainer integration
- Hugging Face model support
- Dataset handling
- Automatic logging

#### 3. Train Reward Models

Learn to train custom reward models:

```bash
python examples/train_reward_model.py
```

**Covers:**
- Neural reward model training
- Reward dataset preparation
- Model evaluation
- Integration with GRPO

#### 4. Symbolic Physics Discovery

Train on toy symbolic constraints with hidden targets in metadata:

```bash
python examples/physics_symbolic_discovery.py --tasks examples/data/symbolic_physics_tasks.jsonl
```

**Covers:**
- Task schema with constraints + derived variables
- Constraint-based symbolic rewards
- Metadata-aware GSPO queries

Evaluate model outputs against constraints:

```bash
python examples/physics_symbolic_evaluate.py \
    --tasks examples/data/symbolic_physics_tasks.jsonl \
    --predictions /path/to/predictions.jsonl
```

### Advanced Training ⭐ NEW!

See **[Advanced Training README](./ADVANCED_TRAINING_README.md)** for detailed guides on:

#### Distributed Multi-GPU Training

Scale training across multiple GPUs:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    examples/distributed_multi_gpu_training.py \
    --model gpt2 \
    --task customer_service
```

#### Custom Reward Functions

Create domain-specific rewards:

```bash
# Test custom rewards
python examples/custom_reward_functions.py

# View available rewards
python examples/custom_reward_functions.py --list-rewards
```

#### Advanced Optimization

Optimize training with cutting-edge techniques:

```bash
python examples/advanced_optimization_techniques.py \
    --model gpt2 \
    --mixed-precision bf16 \
    --compile
```

## 🎯 Domain-Specific Examples

### Customer Service

#### Enhanced Customer Service

Production-ready customer service agent:

```bash
python examples/enhanced_customer_service.py
```

**Features:**
- Multi-turn conversations
- Context management
- Domain-specific rewards
- Empathy and action-oriented responses

#### Production Ready Customer Service

Enterprise-grade implementation:

```bash
python examples/production_ready_customer_service.py
```

**Includes:**
- Error handling and resilience
- Monitoring and logging
- Performance optimization
- Deployment-ready code

### Technical Support

RAG-enabled technical support agent:

```bash
python examples/rag_agent_example.py
```

**Features:**
- Retrieval-Augmented Generation
- Document search
- Technical knowledge base
- Code analysis capabilities

## 🔧 Fine-Tuning Examples

### GSPO (Group Sequence Policy Optimization)

StateSet Agents includes GSPO, a more stable alternative to GRPO.

#### Qwen Models

Fine-tune Qwen models with GSPO:

```bash
python examples/finetune_qwen3_gspo.py \
    --model Qwen/Qwen2.5-7B \
    --task customer_service \
    --use-lora
```

See [QWEN3_FINETUNING_GUIDE.md](../docs/QWEN3_FINETUNING_GUIDE.md)

#### Gemma Models

Fine-tune Google Gemma models:

```bash
python examples/finetune_gemma3_gspo.py \
    --model google/gemma-2-9b-it \
    --task customer_service \
    --use-lora
```

See [GEMMA3_FINETUNING_GUIDE.md](../docs/GEMMA3_FINETUNING_GUIDE.md)

#### Llama Models

Fine-tune Meta Llama models:

```bash
python examples/finetune_llama3_gspo.py \
    --model meta-llama/Llama-3.2-3B \
    --task customer_service \
    --use-lora --use-4bit
```

#### Mistral Models

Fine-tune Mistral models:

```bash
python examples/finetune_mistral_gspo.py \
    --model mistralai/Mistral-7B-v0.1 \
    --task customer_service \
    --use-lora
```

#### Code Assistant

Fine-tune for code generation:

```bash
python examples/finetune_code_assistant.py
```

**Features:**
- Code-specific reward functions
- Syntax validation
- Multi-language support

## 🎨 Framework Showcases

### GRPO Showcase

Comprehensive demonstration of GRPO capabilities:

```bash
python examples/grpo_showcase.py
```

**Demonstrates:**
- Multi-turn trajectory generation
- Group advantage computation
- Policy gradient updates
- Value function training
- Reward shaping

### Enhanced Framework Showcase

Advanced framework features:

```bash
python examples/enhanced_framework_showcase.py
```

**Includes:**
- Circuit breaker patterns
- Memory monitoring
- Type safety features
- Error handling

## 🔌 API Examples

See **[API Examples README](./API_EXAMPLES_README.md)** for complete API documentation.

### API Client (Async)

Full-featured async client:

```bash
python examples/api_client_example.py
```

### API Client (Simple)

Synchronous client for quick integration:

```bash
python examples/api_client_simple.py
```

### Interactive Chatbot

CLI chatbot using the API:

```bash
python examples/interactive_chatbot.py
```

## 🧪 Experimental Examples

### HPO (Hyperparameter Optimization)

Automatic hyperparameter tuning:

```bash
python examples/hpo_training_example.py
```

**Features:**
- Optuna integration
- Automatic search space
- Multi-objective optimization
- Best config selection

### Backend Switching

Dynamic backend switching:

```bash
python examples/backend_switch_demo.py
```

**Demonstrates:**
- Stub backend for testing
- Real model backend
- Runtime switching

## 📦 Prerequisites

### Basic Requirements

```bash
pip install stateset-agents
```

### Development Requirements

For all examples:

```bash
pip install stateset-agents[dev]
```

### Optional Dependencies

For TRL integration:
```bash
pip install stateset-agents[trl]
```

For API examples:
```bash
pip install stateset-agents[api]
```

For HPO:
```bash
pip install stateset-agents[hpo]
```

For all features:
```bash
pip install stateset-agents[dev,api,trl,hpo]
```

## 🎯 Examples by Use Case

### I want to...

#### ...get started quickly
→ `hello_world.py` or `quick_start.py`

#### ...train a customer service agent
→ `enhanced_customer_service.py` or `production_ready_customer_service.py`

#### ...train on multiple GPUs
→ `distributed_multi_gpu_training.py` ⭐ NEW!

#### ...create custom reward functions
→ `custom_reward_functions.py` ⭐ NEW!

#### ...optimize training performance
→ `advanced_optimization_techniques.py` ⭐ NEW!

#### ...fine-tune a specific model
→ Choose from:
- `finetune_qwen3_gspo.py`
- `finetune_gemma3_gspo.py`
- `finetune_llama3_gspo.py`
- `finetune_mistral_gspo.py`

#### ...integrate with my application
→ `api_client_example.py` or `api_client_simple.py`

#### ...build a chatbot
→ `interactive_chatbot.py`

#### ...understand GRPO internals
→ `grpo_showcase.py`

#### ...find optimal hyperparameters
→ `hpo_training_example.py`

## 📝 Example Structure

Each example follows this structure:

```python
"""
Example Title

Description of what the example demonstrates.

Requirements:
    - List of dependencies

Usage:
    # How to run the example
    python examples/example_name.py [options]
"""

# Imports
import asyncio
from stateset_agents import MultiTurnAgent
# ...

# Configuration
# ...

# Main functionality
async def main():
    # Example code
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛠️ Running Examples

### Basic Execution

```bash
python examples/<example_name>.py
```

### With Arguments

```bash
python examples/<example_name>.py --help  # See all options
python examples/<example_name>.py --option value
```

### From Python

```python
import asyncio
from examples import example_module

asyncio.run(example_module.main())
```

## 📚 Learning Path

### Beginner

1. `hello_world.py` - Understand basic concepts
2. `quick_start.py` - Run your first real agent
3. `api_client_simple.py` - Integrate with applications

### Intermediate

4. `enhanced_customer_service.py` - Build domain-specific agents
5. `custom_reward_functions.py` - Create custom rewards ⭐
6. `train_with_trl_grpo.py` - Use HuggingFace TRL

### Advanced

7. `distributed_multi_gpu_training.py` - Scale to multiple GPUs ⭐
8. `advanced_optimization_techniques.py` - Optimize performance ⭐
9. `finetune_qwen3_gspo.py` - Fine-tune large models
10. `hpo_training_example.py` - Automated hyperparameter tuning

## 🐛 Troubleshooting

### Common Issues

#### Import Errors

```bash
pip install -e ".[dev]"  # Install from source
```

#### CUDA Out of Memory

See [Advanced Training README](./ADVANCED_TRAINING_README.md#troubleshooting) for memory optimization tips.

#### Slow Training

Check [Performance Tips](./ADVANCED_TRAINING_README.md#performance-tips) for optimization strategies.

## 📞 Support

- **Documentation**: https://stateset-agents.readthedocs.io/
- **Discord**: https://discord.gg/stateset
- **Issues**: https://github.com/stateset/stateset-agents/issues

## 🤝 Contributing

Want to add an example? See [CONTRIBUTING.md](../CONTRIBUTING.md)

## 📄 License

See [LICENSE](../LICENSE) for details.

---

**Made with ❤️ by the StateSet Team**
