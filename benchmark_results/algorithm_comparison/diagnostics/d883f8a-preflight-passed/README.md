# Five-algorithm CUDA preflight — passed

This retained diagnostic ran on 2026-08-27 from harness commit
`d883f8afbcd168519de485ee7aec79837ac1d056` on one NVIDIA GeForce RTX 5080
(CUDA 12.9). It used seed 42, Qwen2.5-0.5B-Instruct and GSM8K at the pinned
manifest revisions, two held-out examples, one policy step, and two
generations per prompt. RunPod pod `s94dg1mqyvzpu6` was terminated after the
archive was retrieved; the post-run account inventory contained zero pods.

All five algorithms emitted measured evidence and a hashed normalized policy:

| Algorithm | Measured completions | Wall time (s) | Samples/s | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|
| GRPO | 4 | 19.808 | 0.202 | 1,347.1 |
| GSPO | 4 | 18.608 | 0.215 | 1,044.4 |
| DAPO | 4 | 19.567 | 0.204 | 969.1 |
| VAPO | 6 | 21.162 | 0.284 | 2,434.5 |
| GEPO | 4 | 19.620 | 0.204 | 1,770.4 |

VAPO includes two value-warmup generations in addition to four policy
generations, so its samples/s value is not a like-for-like quality or policy
step comparison. Baseline and final exact-match scores were both zero on the
two-example diagnostic evaluation set. These tiny results establish live
execution compatibility only; they are not evidence of model improvement.

This is diagnostic evidence, not a publishable algorithm comparison. The
three-seed gate rejects it. Large model/tokenizer artifacts are omitted from
Git; evidence retains their SHA-256 digests and the raw adapter results retain
the original remote paths.
