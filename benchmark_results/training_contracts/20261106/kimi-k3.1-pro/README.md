# Kimi K3.1 Pro metadata audit — 2026-11-06

Audited official checkpoint `moonshotai/Kimi-K3.1-Pro` at immutable revision
`d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`. No model weights were downloaded,
no inference endpoint was called, and no training hardware was provisioned.

`config.json` retains the released configuration. `metadata.json` records its
SHA-256, the official index SHA-256 and stored tensor size, representative
attention targets, quantization details, and explicit training blockers.
The full 59 MB weight index is not duplicated here. Verify its hash against
`https://huggingface.co/moonshotai/Kimi-K3.1-Pro/resolve/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/model.safetensors.index.json`.

The index contains 768 language attention modules matched by the adapter
selector, including KDA projections and the DSA indexer's `wq_b`, `wk`, and
`weights_proj`. Vision, routed experts, and non-attention modules are excluded.
These target identities do not establish that the released MXFP4 experts can
be loaded for differentiable training.

The pinned repository exposes configuration/tokenizer/processor code, without
a model implementation or standard chat template. K3.1's `kimi_k31` architecture
must not be substituted with K3's `kimi_k3` implementation. Hosted HTTP behavior
is tested with mocked transport in `tests/api/test_kimi_k31.py`; live inference
and training are unverified.
