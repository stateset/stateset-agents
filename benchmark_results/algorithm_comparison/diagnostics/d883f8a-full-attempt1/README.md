# Five-algorithm full matrix — attempt 1

This retained diagnostic ran on 2026-08-27 from harness commit
`d883f8afbcd168519de485ee7aec79837ac1d056` on one NVIDIA GeForce RTX 5080
(CUDA 12.9). It executed the pinned Qwen2.5-0.5B-Instruct/GSM8K protocol with
seeds 42, 1337, and 2026, 32 training examples, 16 held-out examples, four
policy steps, batch/group size four, and rotated algorithm order. RunPod pod
`a876v0r1ur61y0` was terminated after the archive was retrieved.

The fail-closed result was 12 accepted evidence documents and three retained
VAPO failures:

| Algorithm | Accepted | Failed | Median wall time (s) | Median samples/s | Median peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| GRPO | 3 | 0 | 100.320 | 0.638 | 3,432.9 |
| GSPO | 3 | 0 | 102.369 | 0.625 | 2,120.4 |
| DAPO | 3 | 0 | 237.389 | 0.270 | 5,149.6 |
| VAPO | 0 | 3 | — | — | — |
| GEPO | 3 | 0 | 242.012 | 0.264 | 10,097.2 |

Every VAPO seed completed critic warmup and then OOMed during the policy
update at approximately 15.28/15.47 GiB used. The update retained every
prompt's policy and value forward graph until a single final backward call.
The follow-up preserves one optimizer step and the identical mean objective,
but backpropagates each normalized prompt contribution immediately so those
graphs do not coexist.

The accepted runs showed non-null evaluation behavior, but this incomplete
roster is not a publishable comparison and must not be used to rank
algorithms. Large model/tokenizer artifacts are omitted from Git; accepted
evidence retains their SHA-256 digests and raw adapter results retain their
original remote paths.
