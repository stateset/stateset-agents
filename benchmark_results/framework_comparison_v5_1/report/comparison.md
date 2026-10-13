# Measured framework comparison

> Descriptive results only. Every row uses the same protocol, model, data,
> task, and hardware. This report does not assign subjective feature scores.

- Protocol: `stateset-trl-gspo-shootout-v5.1`
- Harness commit: `b006aabb1e46788926acfdce28af030c32199ac1`
- Model: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Task/data: `gsm8k` at `740312add88f781978c0658806c59bc2815b9866`
- Hardware: 1× NVIDIA L40S (CUDA 12.8)
- Shootout manifest SHA-256: `01a21333def8c264d69e529db3537bae94790a68ec0b81fbd640e40b5d8fa058`

| Framework | Version | Seeds | Samples/s | Wall clock (s) | Peak VRAM (MiB) | Baseline | Final | Improvement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stateset-agents-gspo | 0.64.2 | 3 | 0.993 ± 0.028 | 3225.790 ± 90.764 | 10492.440 ± 2.105 | 0.293 ± 0.000 | 0.314 ± 0.015 | 0.021 ± 0.015 |
| trl | 1.12.0 | 3 | 0.999 ± 0.012 | 3204.361 ± 36.910 | 10752.897 ± 0.226 | 0.293 ± 0.000 | 0.311 ± 0.019 | 0.018 ± 0.019 |

## Interpretation boundary

The table establishes results only for the protocol and hardware above.
It is not evidence of ecosystem maturity, developer experience, or
performance on other models and clusters.
