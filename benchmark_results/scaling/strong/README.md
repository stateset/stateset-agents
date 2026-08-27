# Fixed-work strong-scaling evidence

This directory retains the 2026-08-27 three-seed 1/2/4/8-GPU strong-scaling
matrix produced by harness commit
`f0b90809b1dd5d75d2249a1237a371e87ff6a81b` on one host with eight NVIDIA
GeForce RTX 5080 GPUs.

Every topology executes six optimizer updates over the same 196,608-sample
effective global batch per update. The one-GPU reference uses 96 microbatches;
2, 4, and 8 GPUs use 48, 24, and 12 microbatches per rank. DDP synchronizes the
final microbatch of each update. The v3 validator recomputes this partition and
checks that throughput multiplied by measured wall time equals the declared
work.

The retained JSON report passed the predeclared monotonic-throughput and 50%
minimum-efficiency gate. Saved model binaries are represented by SHA-256
digests in each evidence document and are not committed.

Scope: this is a real BF16 group-policy optimization workload and a measurement
of StateSet's single-host DDP path. It is not an LLM-quality result, rollout
serving benchmark, or multi-node scaling claim.
