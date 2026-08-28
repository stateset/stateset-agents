# Failed fixed-global-batch scaling diagnostic

This complete three-seed 1/2/4/8-GPU matrix is intentionally retained because
it failed the publication gate. With a fixed global batch and synchronization
on every optimizer step, throughput decreased as GPUs were added; 8-GPU
efficiency was 2.9% on the eight-RTX-5080 PCIe topology.

The result motivated an explicit weak-scaling protocol with a fixed per-device
batch and accumulated DDP synchronization. It must not be presented as a
passing strong-scaling result, and the later weak-scaling pass does not erase
or supersede it.

- Harness commit: `f6e74789bdf9ac535922079d1132335e74b9d07f`
- Workload digest: `57c3cf032f31fee46619ee646b826260e7719aebd09cc189c6778a529f5b9ba1`
- Hardware: eight NVIDIA GeForce RTX 5080 GPUs, CUDA 12.9, PyTorch 2.8.0
