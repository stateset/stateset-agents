# Measured framework comparison

> Descriptive results only. Every row uses the same protocol, model, data,
> task, and hardware. This report does not assign subjective feature scores.

- Protocol: `stateset-trl-grpo-shootout-v2`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`
- Task/data: `gsm8k` at `740312add88f781978c0658806c59bc2815b9866`
- Hardware: 1× NVIDIA A40 (CUDA 12.8)

| Framework | Version | Seeds | Samples/s | Wall clock (s) | Peak VRAM (MiB) | Baseline | Final | Improvement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stateset-agents | 0.50.0 | 3 | 0.530 ± 0.133 | 1506.109 ± 339.620 | 3450.458 ± 3.236 | 0.172 ± 0.000 | 0.167 ± 0.024 | -0.005 ± 0.024 |
| stateset-agents-gspo | 0.50.0 | 3 | 0.492 ± 0.089 | 1595.422 ± 301.619 | 2692.615 ± 181.650 | 0.172 ± 0.000 | 0.172 ± 0.000 | 0.000 ± 0.000 |
| trl | 1.9.1 | 3 | 0.566 ± 0.109 | 1387.320 ± 243.502 | 3450.396 ± 3.342 | 0.172 ± 0.000 | 0.172 ± 0.016 | 0.000 ± 0.016 |

## Interpretation boundary

The table establishes results only for the protocol and hardware above.
It is not evidence of ecosystem maturity, developer experience, or
performance on other models and clusters.
