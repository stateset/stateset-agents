# Source-bound SFT and serialized FP8 contracts — 2026-10-15

Muse/frontier SFT previously read dataset paths separately for validation,
hashing and training. A file replacement during setup could bind one file's
hash to another file's examples. Both paths now capture each split once and
parse, validate, check overlap and hash those captured bytes. Receipts retain
raw file hashes and normalized conversation hashes. This is a per-process,
per-file guarantee; cross-rank input agreement remains a separate open gate.

The race regression replaces a valid input with invalid JSON while the tokenizer
loads. Training still receives the original examples and the saved receipt
retains their correct hash. Snapshot tests cover raw versus semantic identity,
ID-independent overlap, Unicode line separators, CRLF, malformed data and hash
metadata isolation. Torch <2.7 is rejected before training-backend imports.

The new FP8 contracts serialize generated tiny GLM-5.3 and GLM-5.3-Flash weights,
using FP8 attention plus individually named expert projections and scale grids.
Their actual native loaders dequantize all 40/41 quantized tensors, merge experts,
resolve Flash's forget-gate namespace, and match every tensor against an exact
BF16 reference with no missing/unexpected/mismatched keys. Both complete a real
adapter optimizer update, retain frozen parameters, and export/reload adapters
with identical logits. No upstream quantizer operation is mocked.

Buildr job `62d7a6d2aa80` passed:

- **1,700** targeted regressions, **4 skips**.
- **105** native/tokenizer/FP8 contracts, no skips.
- **104** isolated installed-wheel contracts, no skips; one unpackaged-example
  test was deselected.
- Global Ruff, dirty-source Black/isort, hygiene, release governance, stable API,
  and type checking across **397** packaged modules.

Counts overlap and do not constitute a fresh full-repository test run. The
optional runtime dependency closure is satisfied; exact versions are in
`summary.json`. `verification.json` retains commands, outcomes and the complete
source hash inventory. Every Python file in the wheel matches that inventory.
`input-sha256.json` also binds the principal code, tests and dependency metadata.
JUnit and native per-model reports retain exact outcomes and scope.

The unpublished local 0.66.1 wheel SHA-256 is
`bfd3ee07ca0546a3094a471c818e7b3a386fa0e3220a4e753ff4b999873b407e`.

**Scope:** CPU mechanics with generated FP8/BF16 weights and actual cached
GLM/Qwen tokenizers. No published model weights, GPU execution, optimized GPU
kernels, multi-rank FSDP/EP, representative quality or scaling is certified.
The MiMo, Kimi K3 and DeepSeek V4.1 released-layout blockers remain unchanged.

Run the focused contracts in the optional frontier/Muse runtime:

```bash
pytest -n0 -o addopts='' tests/unit/test_text_sft.py \
  tests/unit/test_frontier_models.py tests/unit/test_muse_spark.py \
  tests/integration/test_frontier_fp8.py
```

Set `STATESET_REQUIRE_FRONTIER_RUNTIME=1` to require the native Transformers
runtime rather than allowing a missing optional runtime to skip FP8 contracts.
