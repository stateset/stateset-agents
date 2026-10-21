# Muse Spark Open integration evidence — 2026-10-15

Native CPU verification completed: **45 Muse-specific tests passed** as part
of the 94-contract native run. A reduced, randomly initialized native Muse
Spark model with real routed experts and vision/audio towers completed
backward/optimizer updates, language-only PEFT adapter export/reload, and a real
TRL SFTTrainer update with assistant-only labels. Encoder parameters remained
frozen. `native_adapter.json` and `native_sft_trainer.json` retain actual metrics.

The template check uses the public native template with a synthetic vocabulary;
it does not certify the checkpoint tokenizer's actual IDs. The configured
Hugging Face account received HTTP 403 for the pinned checkpoint configuration.
**No released Muse tokenizer or model weights were loaded. Full-size GPU SFT/RL,
gated access and distributed FSDP/EP remain unverified.**

The shared final evidence records 1,691 targeted regressions (4 skips), all 397
packaged modules type-checked, 94 native/tokenizer contracts, and 93 isolated
installed-wheel contracts. These counts overlap and include the frontier roster;
they are not all Muse-specific or a new full-repository test run. One wheel test
requiring an unpackaged example was deselected. The installed wheel's complete
Python source inventory matches the verified checkout.

`verification-summary.json`, `verification.json`, `source-verification.json`,
`dependency-closure.json`, JUnit reports and wheel-native reports bind the exact
source/runtime/artifact scope. Optional Torch now requires 2.7+ for native FSDP;
full source verification was reused only after checking identical Python hashes.
The local unpublished 0.66.1 wheel SHA-256 is
`71aa1a79e3ed68e1f57231fa9902ccb3354817a62d1d4773ab58c9d69650de70`.

Follow-up source-bound dataset and serialized FP8 verification is retained in
[the current evidence directory](../text-sft-snapshot-fp8/README.md). It binds the
updated Python source to 1,700 passing regressions, 105 native contracts and
104 installed-wheel contracts. The initial integration reports above remain
historical evidence for their recorded source hashes.
