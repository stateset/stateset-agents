# Frontier open-weight fine-tuning

StateSet records all seven requested model families, using immutable official
checkpoint revisions audited on 2026-10-15, plus the distinct DeepSeek
V4.1 Flash 1017 release audited on 2026-10-17. MiMo has separate RL and MOPD entries.
Registration and a valid attention weight name do not prove that a released
checkpoint can train. None of these full-size checkpoints has completed a
StateSet GPU training verification in this integration.

| Model / preset | Official checkpoint | Training path and current limit |
| --- | --- | --- |
| MiMo-V2.6-Pro / `mimo-v2.6-pro` | [XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) | Blocked: mixed MXFP4 expert / FP8 attention layout needs a validated converter or training loader. |
| MiMo MOPD / `mimo-v2.6-pro-mopd` | [XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) | Same packed-layout blocker. This is a distinct checkpoint. |
| GLM-5.3 / `glm5.3` | [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | Native `glm_moe_dsa` text SFT implementation; FP8 weights explicitly dequantized to BF16. Full-size run unverified. |
| Kimi K3 / `kimi-k3` | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | Blocked: exact `mxfp4-pack-quantized` expert format and custom KDA backward kernels need a validated training loader. |
| GLM-5.3-Flash / `glm5.3-flash` | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | Native `glm5_next` text SFT implementation; explicit FP8 → BF16 dequantization. Full-size run unverified. |
| DeepSeek V4.1 Flash / `deepseek-v4.1-flash` | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Blocked: published inference tensor layout needs a differentiable CED/CSA2 loader and weight conversion. No native `deepseek_v41` implementation exists in the audited Transformers 5.20.0 release. |
| DeepSeek V4.1 Flash 1017 / `deepseek-v4.1-flash-1017` | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) | Registered separately. Blocked: custom packed inference layout needs a differentiable training loader; this release also lacks a native tokenizer chat template and needs a reviewed model-specific codec. |
| Qwen3.8 2.4T A95B / `qwen3.8-2.4t-a95b` | [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | Native `qwen3_5_moe_text` BF16 text SFT implementation. Full-size run unverified. |
| Qwen3.8-Flash-Next / `qwen3.8-flash-next` | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Native `qwen4_exp` BF16 text SFT implementation. Full-size run unverified. |

Several releases use custom weight licenses. The catalog reports the publishers'
license names and links to each pinned repository; do not assume all weights
carry an OSI open-source license or identical commercial terms.

## Inspect and configure without downloading weights

```bash
stateset-agents frontier-models
stateset-agents frontier-models --model kimi-k3
stateset-agents frontier-models --model deepseek-v4.1-flash-1017
stateset-agents frontier-sft --model deepseek-v4.1-flash-1017 --dry-run
stateset-agents frontier-sft --model glm5.3 --dry-run
stateset-agents frontier-sft --model qwen3.8-2.4t-a95b \
  --train-data data/train.jsonl --validation-data data/validation.jsonl \
  --output-dir outputs/qwen38-adapter --write-config qwen38-sft.json
```

These commands return JSON and do not import Torch, provision compute or contact
a provider. `training_path_available` means an implementation exists;
`full_size_training_verified` remains false. Unsupported layouts are visible in
previews and rejected before training data, optional runtime imports or model
weights are accessed. Training calls through the shared Transformers loader also reject these
layouts. Studio validation also blocks known unsupported layouts on local,
RunPod and Modal before reading the dataset or submitting compute; River keeps
its independent exact-model provider check. Inference keeps native quantization settings; it does not silently
expand an FP8 serving model into BF16.

### Direct backend submissions

`RemoteExecutor.validate_spec` checks known frontier loader blockers before
Local, RunPod, Modal, CoreWeave, Nebius or Hugging Face Jobs start a worker,
upload a dataset or allocate compute. These executors report
`uses_packaged_training: true` in their capability JSON. Their generic SFT
worker also rejects native FP8 checkpoints that require the dedicated
`frontier-sft` BF16 dequantization recipe. Both submission and model loading
use the same readiness check.

The guard also applies to `RemoteJobSpec.dry_run`: remote dry runs can still
rent a worker. Use `frontier-models` or `frontier-sft --dry-run` for a local,
free configuration inspection. Unknown models passing admission have no new
training certification. Services with their own training runtime, including
River, keep independent exact-model checks. Harvest admission is exempt from the training-layout
guard; this does not certify inference compatibility for any checkpoint.
Noncanonical job modes on a mutated spec are rejected before dispatch.

## Text dataset and launch

Install `pip install -e '.[frontier]'` in the training environment. The optional
extra requires Torch 2.7+, Transformers 5.20+, TRL 1.14+, PEFT 0.21+ and Accelerate 1.15+.
Use an isolated environment for these versions when maintaining older jobs.

Each JSONL row contains `messages` and an optional `id`:

```json
{"messages":[{"role":"user","content":"Explain the refund policy."},{"role":"assistant","content":"Refunds are available within the stated return window."}]}
```

An optional initial system message is followed by alternating user and assistant
text messages, ending with an assistant reply. Media, tools and unknown fields
are rejected. Training and validation must not contain duplicate conversations,
even under different IDs. Conversations exceeding `max_length` are rejected;
the trainer never truncates away the intended target.

The four native paths require `torchrun`, at least two BF16-capable CUDA GPUs,
and enough aggregate memory for the complete dequantized model plus training
overhead. After configuring the cluster's node count, rendezvous and process count:

```bash
torchrun --nnodes "$NNODES" --nproc-per-node "$GPUS_PER_NODE" \
  --node-rank "$NODE_RANK" --rdzv-id stateset-frontier \
  --rdzv-backend c10d --rdzv-endpoint "$RDZV_ENDPOINT" \
  -m stateset_agents.training.frontier_sft --config qwen38-sft.json --execute
```

Use a unique rendezvous ID for each concurrent run. This is a launch pattern,
not a certified hardware topology. Qwen 2.4T stores about **4.89 TB of BF16 base
tensors** before adapters, activations or optimizer state. Qwen Flash Next stores
about **360 GB** of base tensors, including large n-gram embeddings. FSDP
all-gathers each wrapped unit during execution, so aggregate memory alone is
insufficient: each unit and its activations must fit on a device. Native expert
counts and active parameter counts do not provide a usable single-GPU estimate.

## Training behavior

- Pinned native config, tokenizer and weights; remote code is disabled on the
  four enabled paths. MiMo and Kimi record their remote-code requirement but are
  blocked, so merely listing them never executes downloaded code.
- Load-time FSDP, with per-text-layer wrapping and expert parallelism disabled.
  Inference `device_map="auto"` is not used for distributed training.
- Exact native chat token IDs and offset/prefix-derived assistant loss labels.
  The training dataset is pretokenized once; no TRL template substitution or
  user-token loss is permitted. Qwen 2.4T is thinking-only and binds `reasoning_effort=low`; Flash Next
  binds `enable_thinking=false`. Both templates are exported with the adapter.
- Assistant-token NLL only; router auxiliary loss is explicitly disabled.
  Experts and routing parameters remain frozen.
- Language attention adapters only. Regex targets exclude vision/audio,
  experts, shared experts and MTP. Hybrid attention branches are included.
- FP8 GLM checkpoints are dequantized before adaptation. Generic bitsandbytes
  re-quantization is not substituted for native FP8 or MXFP4 checkpoint layouts.
- Adapter and tokenizer export plus `stateset_training.json` with checkpoint
  identity, source data hash, encoded training hash, runtime versions, template
  hash and actual optimizer step count. Zero-step runs fail.
- Adapter-only saving follows the load-time distributed recipe. Optimizer-state
  resume is unsupported and rejected as an unknown configuration field.

The legacy Kimi starter now uses published KDA/MLA targets and the pinned revision.
Its profiles no longer advertise bitsandbytes QLoRA for the released MXFP4 model.
The legacy command remains a configuration preview; its common model loader
fails with the explicit training blocker on an attempted real run.

## Evidence and remaining work

[Checkpoint audit](../benchmark_results/training_contracts/20261015/frontier-models/metadata.json)
contains revision pins, config/index SHA-256 hashes, storage bytes and actual
representative attention tensor names. The native CPU contracts use tiny random
architectures and a synthetic tokenizer. They cover SFT loss masking, adapter
updates, frozen parameters, export/reload and locally serialized GLM FP8
attention/expert dequantization. They do not establish published-weight loading,
multi-node sharding, GPU kernels or quality improvements.
Full-size and multi-node runs are required before promoting any entry to
training-verified. The three blocked families additionally need their exact
released checkpoint layouts implemented and validated.

The [1017 checkpoint audit](../benchmark_results/training_contracts/20261017/deepseek-v4.1-flash-1017/metadata.json)
pins revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, MIT licensing and
510,286,023,000 bytes of stored tensors. Its config and index match the earlier
Flash metadata; that does not establish identical weight bytes. The release has
no `chat_template.jinja` or inline tokenizer template, and its encoding code has
changed. StateSet does not substitute the earlier template. Its reference
forward runs in inference mode; neither registration nor the upstream inference
example establishes a training backward path. This entry is visible in the TUI
model picker and `finetune models` at `configured` maturity. Provider support
requires separate verification for this exact identifier; the earlier Flash
model's availability does not establish 1017 availability.

Muse Spark and Nemotron have separate paths:
[Muse Spark Open](MUSE_SPARK_OPEN.md), [Nemotron 3.5 VL](NEMOTRON_3_5_VL.md).

The latest [verification summary](../benchmark_results/training_contracts/20261015/sft-rank-consensus/summary.json)
records 1,705 passing targeted regressions (4 skips), 398 type-checked modules,
110 passing native/tokenizer/FP8 contracts and 109 passing isolated-wheel
contracts, plus a two-process installed-wheel consensus test. Counts overlap.
Locally serialized FP8 attention and per-expert
weights now load, dequantize and train through the actual GLM loaders; full-size
published checkpoints and distributed GPU execution remain unverified.

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
