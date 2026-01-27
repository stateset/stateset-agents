# Kimi-K2.5 Integration Examples

This directory contains examples for training and deploying **Moonshot AI Kimi-K2.5** models with the StateSet Agents RL framework.

## 📁 Files Overview

### Core Integration Files
- `kimi_k25_config.py` - Training configuration optimized for Kimi-K2.5
- `finetune_kimi_k25_gspo.py` - Full GSPO training pipeline
- `inference_kimi_k25.py` - Inference/deployment utilities
- `test_kimi_k25_config.py` - Configuration tests

### Documentation
- `KIMI_K25_INTEGRATION.md` - Complete integration guide
- `KIMI_K25_USAGE.md` - Usage examples and best practices

## 🚀 Quick Start

### 1. Basic Training
Train a customer service agent with Kimi-K2.5:

```bash
python examples/finetune_kimi_k25_gspo.py \
  --task customer_service \
  --use-lora \
  --use-vllm \
  --output-dir ./outputs/kimi_k25_customer_service
```

### 2. Training with W&B Logging

```bash
python examples/finetune_kimi_k25_gspo.py \
  --task technical_support \
  --use-lora \
  --use-vllm \
  --wandb \
  --wandb-project kimi-k25-training
```

### 3. Inference with Trained Model

```bash
python examples/inference_kimi_k25.py \
  --checkpoint ./outputs/kimi_k25_customer_service \
  --message "How can I help you today?"
```

## 📊 Supported Tasks

| Task | Description | System Prompt |
|------|-------------|---------------|
| `customer_service` | Customer support conversations | "You are Kimi, a helpful and empathetic customer service representative..." |
| `technical_support` | Technical troubleshooting | "You are Kimi, a knowledgeable technical support specialist..." |
| `sales` | Sales and recommendations | "You are Kimi, a friendly and persuasive sales representative..." |
| `coding_assistant` | Code generation and debugging | "You are Kimi, an expert coding assistant..." |

## 🔧 Configuration Options

### Model Loading
- **LoRA**: Enabled by default (recommended)
  - Rank: 64
  - Alpha: 128
  - Target modules: All attention and FFN layers
- **Quantization**: Optional 4-bit/8-bit support
  - Use `--use-4bit` or `--use-8bit` flags
  - Recommended for GPU memory < 80GB

### Training Parameters
- **Learning Rate**: 3e-6 (optimized for MoE models)
- **Batch Size**: 1 (per device)
- **Gradient Accumulation**: 16 steps
- **Group Size**: 8 (GSPO)
- **Sequence Clipping**: [1.5e-4, 2.5e-4]

### Generation Settings
- **Max Prompt Length**: 8192 tokens
- **Max Completion Length**: 2048 tokens
- **Temperature**: 1.0 (thinking mode), 0.6 (instant mode)
- **Top-p**: 0.95

## 💻 Hardware Requirements

### Minimum Configuration
- **GPU**: 1x A100 40GB or equivalent
- **RAM**: 128GB
- **Disk**: 100GB free space

### Recommended Configuration
- **GPU**: 1x A100 80GB or H100 80GB
- **RAM**: 256GB
- **Disk**: 200GB+ SSD
- **vLLM**: Enabled for faster inference

### Multi-GPU Support
For distributed training:
```bash
torchrun --nproc_per_node=4 examples/finetune_kimi_k25_gspo.py \
  --task customer_service \
  --use-lora
```

## 🎯 Training Optimization Tips

1. **Memory Optimization**
   - Use `--use-vllm` for efficient generation (5-20x faster)
   - Enable quantization with `--use-4bit` if memory limited
   - Reduce `max_completion_length` if not generating long outputs

2. **Speed Optimization**
   - Increase `per_device_train_batch_size` if GPU memory allows
   - Enable gradient checkpointing with `--gradient-checkpointing`
   - Use multi-GPU training for maximum throughput

3. **Quality Optimization**
   - Use custom scenarios in your environment
   - Implement domain-specific reward functions
   - Enable KL penalty with `--beta 0.01` for stability

## 📈 Monitoring and Logging

### W&B Integration
Track training metrics:
```python
# In your training script
trained_agent = await train_with_gspo(
    config=gspo_config,
    agent=agent,
    environment=environment,
    reward_model=reward_model,
)
```

### Key Metrics to Track
- **Loss**: GSPO loss, KL penalty
- **Rewards**: Mean, max, min per episode
- **Advantages**: Group advantage statistics
- **Clipping**: Clipped vs unclippter ratios
- **Learning Rate**: Actual LR schedule
- **Sample Efficiency**: Tokens processed per second

## 🧪 Testing

Run integration tests:
```bash
pytest tests/test_kimi_k25_integration.py -v
```

Test specific components:
```bash
# Test configuration
pytest tests/test_kimi_k25_integration.py::test_kimi_k25_config_creation

# Test training pipeline
pytest tests/test_kimi_k25_integration.py::test_kimi_k25_training_pipeline

# Test inference
pytest tests/test_kimi_k25_integration.py::test_kimi_k25_inference
```

## 🐛 Troubleshooting

### Common Issues

**Issue: Out of Memory**
```bash
# Solution: Enable quantization
python examples/finetune_kimi_k25_gspo.py --use-4bit --use-lora
```

**Issue: Slow Training**
```bash
# Solution: Enable vLLM
python examples/finetune_kimi_k25_gspo.py --use-vllm
```

**Issue: Model Not Loading**
```bash
# Solution: Verify transformers version
pip install 'transformers>=4.57.1'
```

### Getting Help

- Check `KIMI_K25_INTEGRATION.md` for detailed troubleshooting
- Review Apache/vLLM logs for errors
- Monitor GPU utilization with `nvidia-smi`

## 📚 Additional Resources

- [Kimi-K2.5 Model Card](https://huggingface.co/moonshotai/Kimi-K2.5)
- [StateSet Agents Documentation](https://github.com/stateset/stateset-agents)
- [GSPO Paper](https://arxiv.org/abs/2507.18071v2)
- [Moonshot AI Platform](https://platform.moonshot.ai)

## 🤝 Contributing

To contribute new examples:

1. Add your example file to this directory
2. Update this README with usage instructions
3. Add tests to `tests/test_kimi_k25_integration.py`
4. Follow AGENTS.md style guidelines

## 📄 License

This integration follows the same license as StateSet Agents (BUSL-1.1) and Kimi-K2.5 (Modified MIT).