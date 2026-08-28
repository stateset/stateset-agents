# Measured framework comparison

> Descriptive results only. Every row uses the same protocol, model, data,
> task, and hardware. This report does not assign subjective feature scores.

- Protocol: `stateset-trl-grpo-shootout-v1`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`
- Task/data: `gsm8k` at `740312add88f781978c0658806c59bc2815b9866`
- Hardware: 1× NVIDIA A40 (CUDA 12.9)

| Framework | Version | Seeds | Samples/s | Wall clock (s) | Peak VRAM (MiB) | Baseline | Final | Improvement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stateset-agents | 0.42.3 | 3 | 0.326 ± 0.010 | 196.375 ± 6.136 | 3436.694 ± 7.092 | 0.188 ± 0.000 | 0.167 ± 0.072 | -0.021 ± 0.072 |
| trl | 1.9.1 | 3 | 0.327 ± 0.006 | 195.974 ± 3.940 | 3437.032 ± 7.529 | 0.188 ± 0.000 | 0.188 ± 0.108 | 0.000 ± 0.108 |

## Interpretation boundary

The table establishes results only for the protocol and hardware above.
It is not evidence of ecosystem maturity, developer experience, or
performance on other models and clusters.
