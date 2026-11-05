# SFT completion requires a complete export

Industry training previously reported `trained` whenever the trainer returned,
including an empty output directory. Fresh local SFT and the packaged worker now
check their safetensors export, and industry training performs the same check
before recording success. Failed exports retain recoverable files and record a
failed outcome. Preview behavior remains separate and does not require ML imports.

The verifier checks bounded regular JSON metadata, fixed-rank LoRA configuration,
tokenizer configuration, nonempty safetensors payloads, compatible A/B tensor
pairs and matching captured source/parsed-input lineage. It supports a single
file or safe shard references and checks declared shard contents. It rejects
symlinks, special metadata files, ambiguous layouts, missing or corrupt weights,
incomplete pairs, wrong ranks and stale lineage. Fresh exports explicitly use
safe serialization. Producer checks also compare the complete newly written
lineage document, so a failed first lineage write cannot silently succeed.

668 unique selected tests passed, including 28 new export checks. Full-package
mypy passed for 445 files. Ruff, Black, isort, repository hygiene and whitespace
checks passed; Black ran outside the known sandbox worker-IPC restriction.
Existing forwarding tests now create structural export fixtures while keeping
their seed, revision and evaluation assertions. These fixtures are explicitly
not evidence of model training.

The tests include actual tiny CPU LoRA training with locally constructed models
and tokenizers. The retained review rechecks one successful test export, rejects
an actual failed-lineage test export while preserving its weights, and rejects a
corrupted copy. `tiny-cpu-exports.tar.gz` retains both actual test artifacts;
`review_runner.py`, the source archive and package source hashes bind this review.
All earlier evidence bundles were verified without modification.

These are structural consistency checks, not model inference, numerical
finiteness checks, optimizer-update attestation or quality certification. The
fresh generic SFT path uses fixed-rank LoRA safetensors; binary-only legacy and
other provider/native-RL exports retain their own contracts. Keep files stable
during validation; local metadata is not authenticated. The retained tiny CPU
fixtures do not train an industry/River target model. No paid job, target-model
update, held-out quality gain or A+ result is claimed.
