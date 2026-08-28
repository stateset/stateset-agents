# Measured distributed-scaling comparison

Workload digest: `28b356d44d75ca5c824b60354cea30177fe488fd96ffea33c6c49d394ab7443b`
GPU: NVIDIA GeForce RTX 5080
Scaling mode: strong
Default publication gate: monotonic throughput and at least 50% efficiency

| GPUs | Seeds | Samples/s | Speedup | Scaling efficiency | Wall clock (s) | Peak VRAM/GPU (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 336512.383 ± 5186.860 | 1.000× | 100.0% | 3.5 | 589.8 |
| 2 | 3 | 664103.872 ± 1044.358 | 1.973× | 98.7% | 1.8 | 589.8 |
| 4 | 3 | 1237059.818 ± 24604.911 | 3.676× | 91.9% | 1.0 | 589.8 |
| 8 | 3 | 2235220.176 ± 9515.707 | 6.642× | 83.0% | 0.5 | 589.8 |

> Scope: these measurements apply only to the retained workload, model,
> software, GPU, and interconnect configuration.
