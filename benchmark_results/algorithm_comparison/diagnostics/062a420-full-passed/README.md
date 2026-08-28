# Five-algorithm full matrix — passed

This retained raw-run diagnostic accompanies the publication evidence under
[`../../evidence/`](../../evidence/) and the validated
[`../../report/comparison.md`](../../report/comparison.md). It ran on
2026-08-27 from commit `062a420b6cbdb4b532115a6c6500041ac1c68aa2`
on one NVIDIA GeForce RTX 5080 (CUDA 12.9).

All 15 runs passed: GRPO, GSPO, DAPO, VAPO, and GEPO at seeds 42, 1337, and
2026. Every accepted run retained external wall time, actual generated
completion count, peak VRAM, baseline/final held-out scores, exact shared and
algorithm-specific configurations, pinned model/dataset revisions, and a
normalized policy SHA-256 digest. The strict aggregator accepted the complete
roster without exceptions.

RunPod pod `2vb0s0nj933ux9` was terminated after evidence retrieval. The
downloaded archive SHA-256 was
`acc4efdf3d46a3b42572940e7326fc387c3753c8b0409c0fcb92855d7e123366`;
the console-log SHA-256 was
`058d109f0be19a82ea7822dd1cdad0fc5cec87ef58e0a2ddda4a98faab4fe5bd`.
Large model/tokenizer artifacts are omitted from Git; their content digests
remain in the publication evidence.
