# Frontier checkpoint integration evidence — 2026-10-15

Eight exact official checkpoints cover all seven requested model families plus
MiMo MOPD. `metadata.json` records pinned config/index hashes, storage sizes,
license metadata and representative attention names from the complete indexes.
The raw index namespace differs from the native GLM Flash runtime: Transformers
moves `f_a_proj` / `f_b_proj` below `self_attn.forget_gate`. The implemented
adapter regex handles both namespaces; real native tests exercise the latter.

Verification completed through Buildr:

- **1,691 targeted regressions passed, 4 skipped**, with style, hygiene,
  release governance, stable API and all **397 packaged modules** type-checked.
- **94 native/tokenizer contracts passed** in the optional training runtime.
  Four real tiny GLM/Qwen architectures completed SFT optimizer updates,
  preserved frozen parameters and exported/reloaded adapters. Muse's real tiny
  architecture contracts also passed.
- All **four actual pinned GLM/Qwen tokenizers** passed offline, single- and
  multi-turn loss-mask checks; user tokens remain masked and native IDs match.
- **93 contracts passed against the installed wheel** outside the checkout.
  One unpackaged-example test was deselected. Every packaged Python file in
  the wheel matches the verified source SHA-256 inventory exactly.

Counts overlap between stages and are not a unique combined total. This is a
focused integration/regression run, not a fresh full-repository test run.

`verification-summary.json` provides counts and queue handles.
`source-verification.json` binds the broad regressions and type gate to the exact
397-module source. `verification.json` binds the final wheel, requirements and
repeated native contracts to that unchanged source. The later packaging check
raised the optional Torch minimum to 2.7 for native FSDP; the Python source was
unchanged, so it reused the passed type/regression stages with a recorded hash.
`dependency-closure.json` verifies the actual optional runtime's dependencies.
JUnit reports and per-model native reports retain scope and outcomes. Failed
regression/native attempts are retained alongside the final successful reports.

The wheel SHA-256 is
`71aa1a79e3ed68e1f57231fa9902ccb3354817a62d1d4773ab58c9d69650de70`.
It is an unpublished local build of 0.66.1, not a new release.

**No published model weights were downloaded and no GPU training occurred.**
Tiny CPU contracts do not certify FP8 weight loading/dequantization, load-time
FSDP, multi-node execution, optimized GPU kernels or model quality. MiMo Pro,
Kimi K3 and DeepSeek V4.1 Flash retain explicit released-layout loader blockers.
The four GLM/Qwen SFT implementations remain unverified at full size.
See [the guide](../../../../docs/FRONTIER_FINETUNING.md).

Follow-up source-bound dataset and serialized FP8 verification is retained in
[the current evidence directory](../text-sft-snapshot-fp8/README.md). It binds the
updated Python source to 1,700 passing regressions, 105 native contracts and
104 installed-wheel contracts. The initial integration reports above remain
historical evidence for their recorded source hashes.
