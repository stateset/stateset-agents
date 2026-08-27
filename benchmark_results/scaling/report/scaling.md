# Measured distributed-scaling comparison

Workload digest: `e6e9a653604b1503776687dd431c2df65e55dda83e06a9f08f1520d49a55fd78`
GPU: NVIDIA GeForce RTX 5080
Scaling mode: weak
Default publication gate: monotonic throughput and at least 50% efficiency

| GPUs | Seeds | Samples/s | Speedup | Scaling efficiency | Wall clock (s) | Peak VRAM/GPU (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 322573.708 ± 7286.293 | 1.000× | 100.0% | 3.7 | 589.8 |
| 2 | 3 | 672580.944 ± 1308.594 | 2.085× | 104.3% | 3.5 | 589.8 |
| 4 | 3 | 1291535.299 ± 12911.258 | 4.004× | 100.1% | 3.7 | 589.8 |
| 8 | 3 | 2606529.702 ± 45376.834 | 8.080× | 101.0% | 3.6 | 589.8 |

> Scope: these measurements apply only to the retained workload, model,
> software, GPU, and interconnect configuration.
