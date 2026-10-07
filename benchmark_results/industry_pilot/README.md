# Industry fine-tuning GPU pilots

These pilots check the shipped preparation, LoRA training, adapter reload, and
paired reference-evaluation path on real pretrained checkpoints. They use
explicitly synthetic examples and do not certify industry quality, autonomous
task success, business outcomes, or RL performance.

- [2026-10-07 Qwen3.5 2B/4B pilot](20261007/README.md)

Each attempt retains its source/model revisions, dependency versions, provider
record, and failures. Completed model runs retain raw predictions, prepared
data, reference scores, optimizer-step evidence, and adapter hashes. Model
weights and optimizer tensors are excluded from Git.
