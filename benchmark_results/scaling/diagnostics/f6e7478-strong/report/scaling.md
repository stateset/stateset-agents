# Measured distributed-scaling comparison

Workload digest: `57c3cf032f31fee46619ee646b826260e7719aebd09cc189c6778a529f5b9ba1`
GPU: NVIDIA GeForce RTX 5080
Default publication gate: monotonic throughput and at least 50% efficiency

| GPUs | Seeds | Samples/s | Speedup | Scaling efficiency | Wall clock (s) | Peak VRAM/GPU (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 261437.014 ± 3858.402 | 1.000× | 100.0% | 0.2 | 595.2 |
| 2 | 3 | 149113.019 ± 1151.629 | 0.570× | 28.5% | 0.3 | 442.4 |
| 4 | 3 | 111463.751 ± 531.539 | 0.426× | 10.7% | 0.4 | 477.7 |
| 8 | 3 | 59862.329 ± 894.526 | 0.229× | 2.9% | 0.8 | 477.8 |

> Scope: these measurements apply only to the retained workload, model,
> software, GPU, and interconnect configuration.
