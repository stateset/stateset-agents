Muse Spark Open
===============

``stateset-agents muse-spark-open`` configures revision-pinned, assistant-only
text LoRA SFT with Transformers/TRL load-time FSDP and expert parallelism.
It defaults to a preview and validates local JSONL before loading weights.

The separate ``muse-spark-open`` preset in the unified GSPO driver integrates
the native conditional-generation model for text RL. Full-size SFT/RL GPU
conformance remains open; the checkpoint is gated and requires cluster-scale
memory. Load-time sharded SFT saves adapters only, without optimizer resume.

See the
`complete Muse Spark guide <https://github.com/stateset/stateset-agents/blob/main/docs/MUSE_SPARK_OPEN.md>`_
for dependencies, datasets, multi-node launch, checkpoint identity, template
options, adapter output, and verification limits.
