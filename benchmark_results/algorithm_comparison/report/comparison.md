# Measured algorithm comparison

> Descriptive results only. Every row uses the same protocol, model, data,
> task, and hardware. This report does not assign subjective feature scores.

- Protocol: `stateset-five-algorithm-gsm8k-v1`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`
- Task/data: `gsm8k` at `740312add88f781978c0658806c59bc2815b9866`
- Hardware: 1× NVIDIA GeForce RTX 5080 (CUDA 12.9)

| Algorithm | Version | Seeds | Samples/s | Wall clock (s) | Peak VRAM (MiB) | Baseline | Final | Improvement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dapo | stateset-dapo-token-objective-v1 | 3 | 0.261 ± 0.005 | 244.799 ± 4.283 | 5092.068 ± 100.875 | 0.062 ± 0.000 | 0.146 ± 0.036 | 0.083 ± 0.036 |
| gepo | stateset-gepo-group-expectation-objective-v1 | 3 | 0.265 ± 0.005 | 241.127 ± 4.154 | 10019.537 ± 442.036 | 0.062 ± 0.000 | 0.167 ± 0.072 | 0.104 ± 0.072 |
| grpo | trl-grpo-objective-v1 | 3 | 0.627 ± 0.015 | 102.080 ± 2.486 | 3436.694 ± 7.092 | 0.062 ± 0.000 | 0.125 ± 0.125 | 0.062 ± 0.125 |
| gspo | stateset-gspo-sequence-objective-v1 | 3 | 0.599 ± 0.027 | 106.977 ± 4.804 | 2170.469 ± 121.861 | 0.062 ± 0.000 | 0.062 ± 0.000 | 0.000 ± 0.000 |
| vapo | stateset-vapo-value-augmented-objective-v1 | 3 | 0.261 ± 0.001 | 260.527 ± 1.480 | 6078.170 ± 55.470 | 0.062 ± 0.000 | 0.188 ± 0.108 | 0.125 ± 0.108 |

## Interpretation boundary

The table establishes results only for the protocol and hardware above.
It is not evidence of ecosystem maturity, developer experience, or
performance on other models and clusters.
