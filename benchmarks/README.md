# StateSet Agents Benchmarks

This directory contains comprehensive benchmark suites for the StateSet Agents framework.

## Available Benchmarks

### 1. Benchmark Suite (`benchmark_suite.py`)
Complete end-to-end benchmarks for all framework features.

**Usage:**
```bash
# Run all benchmarks
python benchmarks/benchmark_suite.py --all

# Run specific benchmarks
python benchmarks/benchmark_suite.py --latency --throughput

# Generate HTML report
python benchmarks/benchmark_suite.py --all --report html

# Generate JSON report
python benchmarks/benchmark_suite.py --all --report json
```

**Benchmarks Included:**
- Agent latency (stub and real models)
- Agent streaming performance
- Input validation throughput
- Structured output parsing
- Function calling overhead
- Memory system performance
- Evaluation metrics computation
- Batch processing throughput
- Config validation speed

### 2. Performance Benchmarks (`performance_benchmarks.py`)
Detailed GRPO framework performance analysis.

**Usage:**
```bash
# Run all performance benchmarks
python benchmarks/performance_benchmarks.py

# Results saved to benchmark_results/
```

**Benchmarks Included:**
- Agent response generation
- Multi-turn conversation handling
- Concurrent conversation processing
- Reward system computation
- Computational engine performance
- Memory usage patterns
- CPU utilization
- System-level benchmarks

### 3. Algorithm Comparison (`algorithm_comparison.py`)
Compare different RL algorithms (GRPO, GSPO, PPO, DPO).

**Usage:**
```bash
# Compare all algorithms
python benchmarks/algorithm_comparison.py --algorithms all

# Compare specific algorithms
python benchmarks/algorithm_comparison.py --algorithms grpo gspo

# Quick test with stub model
python benchmarks/algorithm_comparison.py --stub --episodes 100

# Custom model and episodes
python benchmarks/algorithm_comparison.py \
  --model gpt2-medium \
  --episodes 1000 \
  --output ./my_results
```

**Metrics Compared:**
- Convergence speed (episodes to 80% reward)
- Final reward achieved
- Training stability (coefficient of variation)
- Sample efficiency (reward per 1K samples)
- Memory usage
- Training time

### 4. Framework Comparison (`framework_comparison.py`)
Compare StateSet Agents against TRL, Ray RLlib, and custom implementations.

**Usage:**
```bash
# Compare all frameworks
python benchmarks/framework_comparison.py --frameworks all

# Compare specific frameworks
python benchmarks/framework_comparison.py --frameworks stateset trl

# Quick test
python benchmarks/framework_comparison.py --stub --samples 100
```

**Metrics Compared:**
- Training speed (samples/second)
- Memory usage
- Setup time and complexity (LOC)
- Final performance
- Features score
- Ease of use score

### 5. Real Performance Benchmarks (`real_performance_benchmarks.py`)
Production-grade performance validation with real metrics.

**Usage:**
```bash
# Run comprehensive benchmarks
python benchmarks/real_performance_benchmarks.py

# Results include:
# - Response time distribution (P50, P95, P99)
# - Throughput measurements
# - Concurrent conversation handling
# - Memory profiling
```

## Quick Start

### Run All Benchmarks (Fast Test)
```bash
# Use stub models for fast validation (~2 minutes)
python benchmarks/benchmark_suite.py --all --report both
python benchmarks/algorithm_comparison.py --stub --episodes 100
python benchmarks/framework_comparison.py --stub --samples 100
```

### Run Production Benchmarks
```bash
# Full benchmarks with real models (~1-2 hours)
python benchmarks/performance_benchmarks.py
python benchmarks/algorithm_comparison.py --model gpt2-medium --episodes 1000
```

## Interpreting Results

### Performance Benchmarks

**Good Performance Indicators:**
- Latency P95 < 200ms for real-time applications
- Throughput > 10 conversations/sec for production
- Memory usage < 30GB for single GPU training
- GPU utilization > 80% during training

**Red Flags:**
- Latency P99 > 1000ms (poor user experience)
- Memory usage growing over time (memory leak)
- GPU utilization < 50% (CPU bottleneck)
- High standard deviation in metrics (instability)

### Algorithm Comparison

**Choosing an Algorithm:**
- **GSPO:** Best for most cases - stable, efficient, performant
- **GRPO:** Good baseline - simple, proven, well-tested
- **PPO:** Use for compatibility with existing PPO code
- **DPO:** Best sample efficiency early, plateaus later

**Convergence Benchmarks:**
- Good: < 400 episodes to 80% reward
- Excellent: < 300 episodes to 80% reward
- Outstanding: < 250 episodes to 80% reward

**Stability Scores (Coefficient of Variation):**
- Excellent: < 5%
- Good: 5-10%
- Moderate: 10-15%
- Poor: > 15%

### Framework Comparison

**Key Decision Factors:**
1. **Multi-turn conversations** → StateSet Agents
2. **Maximum flexibility** → TRL
3. **Massive scale (1000+ nodes)** → Ray RLlib
4. **Quick time-to-production** → StateSet Agents
5. **Research experimentation** → TRL or StateSet

## Reproducibility

All benchmarks use:
- Fixed random seeds (42)
- Consistent hyperparameters
- Standardized environments
- Documented hardware specifications

To reproduce results exactly:
```bash
# Set random seed
export PYTHONHASHSEED=42

# Run with fixed seed
python benchmarks/algorithm_comparison.py \
  --model gpt2 \
  --episodes 1000 \
  --output ./reproducible_results
```

## Hardware Specifications

**Minimum for Benchmarks:**
- CPU: 4 cores
- RAM: 16 GB
- GPU: Not required (use --stub)
- Storage: 10 GB

**Recommended for Real Benchmarks:**
- CPU: 8+ cores
- RAM: 32 GB+
- GPU: NVIDIA A100 40GB or RTX 4090 24GB
- Storage: 50 GB SSD

**Production Benchmarks:**
- CPU: 16+ cores
- RAM: 64 GB+
- GPU: 4x A100 80GB
- Storage: 1 TB NVMe SSD
- Network: InfiniBand for multi-node

## Benchmark Results Storage

Results are saved to:
- `./benchmark_results/` (default)
- JSON format: Machine-readable metrics
- HTML format: Human-readable reports
- Markdown format: Detailed analysis

**Result Files:**
```
benchmark_results/
├── benchmark_results.json          # Full benchmark suite
├── benchmark_results.html          # HTML report
├── algorithm_comparison.json       # Algorithm metrics
├── algorithm_comparison.md         # Algorithm report
├── framework_comparison.json       # Framework metrics
├── framework_comparison.md         # Framework report
└── grpo_benchmark_results_*.json   # Timestamped runs
```

## Continuous Integration

To run benchmarks in CI/CD:

```yaml
# .github/workflows/benchmarks.yml
name: Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e ".[dev,benchmarks]"
      - name: Run fast benchmarks
        run: |
          python benchmarks/benchmark_suite.py --all --stub
          python benchmarks/algorithm_comparison.py --stub --episodes 10
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmark_results/
```

## Custom Benchmarks

Create your own benchmarks:

```python
from benchmarks.benchmark_suite import benchmark

@benchmark(name="My Custom Benchmark", iterations=100)
def my_benchmark():
    # Your code here
    result = expensive_operation()
    return {"metric": result}

# Run it
result = await my_benchmark()
print(f"Average time: {result.avg_time_ms}ms")
```

## Troubleshooting

### Out of Memory
```bash
# Use smaller model
python benchmarks/algorithm_comparison.py --model gpt2-small

# Reduce episodes
python benchmarks/algorithm_comparison.py --episodes 100

# Use stub mode
python benchmarks/algorithm_comparison.py --stub
```

### Slow Performance
```bash
# Use stub mode for development
python benchmarks/benchmark_suite.py --all --stub

# Reduce iterations
python benchmarks/benchmark_suite.py --latency --iterations 10
```

### Import Errors
```bash
# Ensure proper installation
pip install -e ".[dev,benchmarks]"

# Or install specific dependencies
pip install numpy psutil
```

## Contributing

To add new benchmarks:

1. Follow existing benchmark patterns
2. Include docstrings and type hints
3. Add command-line arguments
4. Generate both JSON and readable reports
5. Document in this README
6. Add example usage
7. Submit PR with benchmark results

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## References

- Full benchmark documentation: [docs/BENCHMARKS.md](../docs/BENCHMARKS.md)
- Methodology: [docs/BENCHMARKS.md#benchmark-methodology](../docs/BENCHMARKS.md#benchmark-methodology)
- Hardware specs: [docs/BENCHMARKS.md#hardware-specifications](../docs/BENCHMARKS.md#hardware-specifications)

## Support

Questions about benchmarks?
- Open an issue: https://github.com/stateset/stateset-agents/issues
- Join Discord: https://discord.gg/stateset
- Email: benchmarks@stateset.com

---

**Last Updated:** December 9, 2025
**Maintained By:** StateSet Team
