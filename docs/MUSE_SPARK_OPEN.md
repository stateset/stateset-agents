# Muse Spark Open

StateSet integrates `meta-models/Muse-Spark-Open-816B-A42B` through its native
`MuseSparkForConditionalGeneration` architecture. The default checkpoint revision
is `0fb02478b02565b2ce7fc57e6051c91172774ac9`. This model is separate from
Muse Glimmer; the Glimmer command remains available.

The dedicated `muse-spark-open` command prepares **assistant-only text LoRA
SFT** using Transformers' load-time FSDP and expert parallelism through TRL.
Adapters target language attention Q/K/V/output projections. Perception/audio
encoders, the expert tensors, and the rest of the base remain frozen. It accepts
plain text conversations; this path does not ingest image, video, audio, or
structured tool messages. The underlying model's additional modalities need
their native processor and a separate training integration.

## Install and prepare

```bash
pip install -e '.[muse-spark]'
```

The optional runtime requires Transformers 5.20+, TRL 1.14+, PEFT 0.21+, and
Accelerate 1.15+. Core imports and dry runs do not import these libraries.
The checkpoint is gated: obtain access on its
[Hugging Face page](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)
and configure Hugging Face authentication in the training environment. Review
the linked Muse Model License, including its attribution requirements. The
account checked during integration received HTTP 403 for the pinned config;
StateSet has not downloaded or trained the released weights.

Use JSONL conversations with alternating user/assistant text and an optional
initial system message:

```json
{"messages":[{"role":"user","content":"Explain this refund policy."},{"role":"assistant","content":"Refund delivered orders within the stated return window."}]}
```

Rows may have an `id`. Unknown fields, missing/empty content, unsupported media,
and invalid message order fail before model loading. Validation conversations
must be disjoint from training even if their IDs differ. After loading only the
tokenizer, StateSet checks every example for assistant targets and rejects any
sequence above `max_length` before allocating model weights. It does not silently
truncate targets. The native chat template supplies assistant masks; the saved
tokenizer fixes `reasoning_effort` and `current_date` so rendering is repeatable.

## Configure and launch SFT

```bash
# Safe configuration and data validation; no weight downloads or paid calls.
stateset-agents muse-spark-open --train-data data/train.jsonl \
  --validation-data data/validation.jsonl --output-dir outputs/muse-spark \
  --write-config muse-spark.json
stateset-agents muse-spark-open --config muse-spark.json --check-data

# Illustrative multi-node topology, not a StateSet-certified hardware size.
# Set NODE_RANK, MASTER_ADDR and MASTER_PORT in the cluster environment.
torchrun --nnodes=8 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m stateset_agents.training.muse_spark --config muse-spark.json --execute
```

The JSON includes `max_steps`, `max_length`, `learning_rate`,
`gradient_accumulation_steps`, `expert_parallel_size`, adapter rank/alpha,
checkpoint intervals, preprocessing workers, seed, reasoning effort, and a fixed
date. Unknown settings fail validation. `WORLD_SIZE` must be at least two and
divisible by `expert_parallel_size` (default eight). FSDP spans the world and
experts use the requested expert-parallel size. The implementation does not use
an inference `device_map` in this distributed training path. Packing is disabled
to keep loss boundaries explicit.

The default BF16 base has approximately 1.64 TB of tensor data according to the
published safetensors metadata, before activations, adapters, buffers, and
training overhead. Budget all 816B parameters, independently of the 42B active
count. The model card's roughly 2 TB BF16 and 700 GB NVFP4 figures concern
inference; they are not StateSet training measurements. The NVFP4 model is a
separate deployment artifact and is not selected by this BF16 SFT command.

The [upstream TRL recipe](https://github.com/huggingface/trl/tree/main/examples/sft_muse_spark_open)
uses load-time sharding. That path does not support optimizer-state resume:
StateSet saves adapters only and deliberately exposes no resume option.
`stateset_training.json` records the exact base revision, dataset hashes,
configuration, committed global step, and metrics. The adapter and matching
tokenizer are saved without publishing the base weights. Reload against that
same BF16 revision using PEFT in a sufficiently sized environment.

## Native text RL preset

The unified GSPO driver also exposes an attention-only text preset:

```bash
python examples/finetune_gspo.py --model muse-spark-open --dry-run
```

It pins the same revision on both the agent and training configuration, uses the
existing composite model loader, and excludes vision/audio adapter targets.
The preset fixes `reasoning_effort="low"` and `current_date="2026-10-15"` through
`AgentConfig.chat_template_kwargs`. Change these explicitly for a new experiment.
Native recipient headers are preserved during decoding, and only `to=user`
content reaches the environment. Private reasoning and commentary remain outside
the final answer, while sampled token IDs and probabilities retain the complete
generation for RL scoring. Muse streaming buffers generation before emitting
final-answer chunks. Failed native template rendering propagates as an error.
It is a native text-RL integration, with full-size GPU conformance still open.
It does not use the distributed SFT loading recipe above; do not infer that the
SFT topology or upstream throughput applies to native GSPO. Generic inference
quantization is disabled by default. No River hosting access is implied.

On the shared workstation, route expensive installs, training, checks, and tests
through Buildr with their complete original command after `--`.

Official references: [model card](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B),
[Meta developer guide](https://dev.meta.ai/resources/blog/build-with-muse-spark-open),
[Transformers architecture](https://huggingface.co/docs/transformers/main/model_doc/muse_spark).

The distributed extra requires Torch 2.7 or newer for native load-time FSDP.

Dataset receipts bind validation and training to one captured read per split.
Raw file hashes and normalized conversation hashes are computed from the same
bytes; edits during tokenizer/model initialization cannot substitute different
training examples or provenance. Torch 2.7 is checked before backend imports.

Before loading weights, every rank must pass the distributed SFT identity
consensus. It binds dataset hashes/counts, tokenization and encoded loss targets,
training settings, checkpoint identity, runtime versions and critical StateSet
source hashes. `stateset_training.json` retains the agreed `rank_consensus`
receipt. Node-local paths may differ; input bytes and training identities must
agree. The guard requires the initialized process group and cannot fall back to
local success. CPU/Gloo consensus tests do not certify GPU/FSDP training.

The [input-consensus evidence](../benchmark_results/training_contracts/20261015/sft-rank-consensus/summary.json)
includes real two-process rejection/recovery checks against both source and the
installed wheel. Each rank runs 17 scenarios, including 15 rejected identities;
both ranks report identical outcomes and can agree after the rejected calls.
