# Five-algorithm CUDA preflight — attempt 1

This retained diagnostic ran on 2026-08-27 from harness commit
`c537209ff67863f16fdc749996ab00a617238284` on one NVIDIA GeForce RTX 5080
(CUDA 12.9). It used seed 42, Qwen2.5-0.5B-Instruct at the pinned manifest
revision, two held-out GSM8K examples, one intended policy step, and two
generations per prompt. RunPod pod `jg63olthyjiuc0` was terminated after the
archive was retrieved; the post-run account inventory contained zero pods.

This is diagnostic evidence, not a publishable algorithm comparison. The
three-seed gate rejects it.

| Algorithm | Result | Live finding |
|---|---|---|
| GRPO | Failed before training | The reduced batch of 1 was not divisible by 2 generations. The full manifest's 4/4 shape is valid; the retry uses 2/2. |
| GSPO | Passed | 2 measured completions, 18.670 s external wall time, 1,032.6 MiB peak VRAM, hashed normalized policy. |
| DAPO | Passed | 2 measured completions (including dynamically filtered groups), 17.916 s external wall time, 968.9 MiB peak VRAM, hashed normalized policy. |
| VAPO | Failed during value warmup | A bf16 policy hidden state reached an fp32 value head without an explicit cast. The follow-up keeps the critic fp32 and casts hidden states differentiably. |
| GEPO | Failed before model load | Importing GEPO required optional `wandb` even with logging disabled. The follow-up gates W&B only when requested. |

The run demonstrates that the orchestrator continued after failures and
retained all five run directories. Large model/tokenizer artifacts are omitted
from Git; accepted evidence preserves their SHA-256 digests and the raw adapter
results preserve the original remote paths.
