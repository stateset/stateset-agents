# Fine-tune Nemotron 3.5 Super VL and Bolt

StateSet provides a dedicated image-conditioned supervised fine-tuning path for
both NVIDIA checkpoints:

| CLI command | Fine-tuning checkpoint |
|---|---|
| `nemotron-3-5-super-vl` | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` |
| `nemotron-3-5-bolt` | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16` |

These commands use Transformers and PEFT with BF16 base weights and LoRA on
language attention projections. Images remain images throughout processing;
only assistant output contributes to the loss. The vision tower, projector,
Mamba projections, and routed experts remain frozen. Mamba fused kernels can
access projection weights directly, bypassing LoRA forward hooks, so this path
deliberately targets attention only. This is SFT, a starting point for domain
adaptation before RL. The existing `nemotron-3-5` Lightning GSPO command remains
a separate text training path.

Both checkpoints and their custom Python code are pinned to immutable revisions
in `stateset_agents.training.nemotron_vl`. Loading them executes NVIDIA's remote
model/processor code. Explicit revision overrides must be full commit SHAs.

## Environment

Use a CUDA environment with BF16 support and enough GPU and host memory:

```bash
pip install -e '.[nemotron-vl]'
# Install CUDA/PyTorch-compatible Mamba kernels in that training environment.
pip install mamba-ssm causal-conv1d --no-build-isolation
```

Bolt's **24GB** checkpoint is a quantized inference artifact. It is not the BF16
fine-tuning memory requirement. Super VL has about 120B total parameters and Bolt
about 74B; planning memory from their active parameter counts would undercount
the base weights. This implementation loads BF16 weights into host memory before
Trainer wraps the model; multi-process runs may require a full host copy per
rank. Plan host RAM, download disk, activations, and sharding accordingly.

NVIDIA also publishes a distinct, optimized
[Super VL NeMo AutoModel LoRA recipe](https://docs.nvidia.com/nemo/automodel/model-coverage/omni/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16).
Its reported GPU memory and throughput are specific to that implementation.
StateSet's Transformers/PEFT and FSDP paths have deterministic CPU contract
coverage, including real adapter updates, but full-size GPU training and memory
usage have **not** been validated here.

The opt-in processor contract downloads only the pinned NVIDIA processors and
tokenizers and checks real single-image, multi-image, and multi-turn batches:

```bash
STATESET_TEST_HF_PROCESSORS=1 pytest -n0 -o addopts='' tests/integration/test_nemotron_vl_processors.py
```

## Local image/chat dataset

Create `data/train.jsonl` with one conversation per line:

```json
{"id":"receipt-1","messages":[{"role":"user","content":[{"type":"image","image":"images/receipt-1.png"},{"type":"text","text":"Extract the merchant, total, and currency as JSON."}]},{"role":"assistant","content":"{\"merchant\":\"Example Store\",\"total\":20.0,\"currency\":\"USD\"}"}]}
```

Image paths resolve relative to the JSONL directory, or `--image-root` when
provided. They must stay within that directory, including after symlink
resolution. Images decode lazily to RGB. Use a separate validation JSONL file
with held-out examples; splitting train and validation is your responsibility.
Multiple images and alternating user/assistant turns are supported, with an
optional initial system message. Assistant content can be text or text blocks.
NeMo's `conversation` key is accepted as an alternative to `messages`.
Within a multi-image conversation, images receive centered black padding to a
common canvas before NVIDIA's processor runs. This preserves aspect ratios and
all original pixels while satisfying the released processor/projector's equal
shape requirement; no image is cropped. Single-image examples use the native
processor dimensions without this padding.

This first path accepts local images and text. Video, audio, tool messages,
remote media URLs, extra fields, missing images, and malformed roles fail
validation. Per-device batch size is one; increase gradient accumulation for
larger effective batches. Examples exceeding `max_length` fail rather than
truncating visual tokens or training without assistant targets.

## Configure, validate, and train

```bash
# Both commands default to dry-run; neither downloads weights here.
stateset-agents nemotron-3-5-super-vl --train-data data/train.jsonl \
  --validation-data data/validation.jsonl --output-dir outputs/super-vl \
  --write-config super-vl.json
stateset-agents nemotron-3-5-bolt --train-data data/train.jsonl \
  --validation-data data/validation.jsonl --output-dir outputs/bolt \
  --write-config bolt.json

stateset-agents nemotron-3-5-super-vl --config super-vl.json --check-data
stateset-agents nemotron-3-5-bolt --config bolt.json --check-data

# Execute on a machine with sufficient GPU/host memory.
stateset-agents nemotron-3-5-bolt --config bolt.json --no-dry-run
```

JSON settings include `max_steps`, `max_length`, `learning_rate`, `lora_rank`,
`lora_alpha`, `gradient_accumulation_steps`, `seed`, `save_steps`, `eval_steps`,
`gradient_checkpointing`, `revision`, and `resume_from_checkpoint`. Unknown
settings fail validation. Checkpoints retain Trainer optimizer/scheduler state;
set `resume_from_checkpoint` to a Trainer checkpoint directory to resume.

For distributed FSDP, write a configuration with `--fsdp` and launch the module:

```bash
stateset-agents nemotron-3-5-super-vl --config super-vl.json --fsdp \
  --write-config super-vl-fsdp.json
torchrun --standalone --nproc-per-node=8 \
  -m stateset_agents.training.nemotron_vl --config super-vl-fsdp.json --execute

stateset-agents nemotron-3-5-bolt --config bolt.json --fsdp \
  --write-config bolt-fsdp.json
torchrun --standalone --nproc-per-node=8 \
  -m stateset_agents.training.nemotron_vl --config bolt-fsdp.json --execute
```

Eight processes above illustrate launching; they are not a certified sizing
claim. FSDP2 preserves parameters to handle mixed frozen/LoRA weights, wraps
the appropriate Nemotron block class, and uses activation checkpointing when
enabled. Multi-process replication without FSDP is rejected.

On the shared workstation, submit expensive commands through Agent Builds,
keeping the full original command after `--` and requesting resources suitable
for the chosen training machine.

Python callers can use the same implementation:

```python
from stateset_agents.training import NemotronVLConfig, train_nemotron_vl

config = NemotronVLConfig(
    model="bolt", train_data="data/train.jsonl", output_dir="outputs/bolt",
    max_steps=100, gradient_accumulation_steps=8,
)
# Requires the CUDA training environment described above.
result = train_nemotron_vl(config)
```

## Adapter output and reload

The output directory contains PEFT adapters, the matching processor/tokenizer,
Trainer checkpoints, and `stateset_training.json` with the base model, exact
revision, configuration, committed global step, and training metrics. Training
does not automatically publish or merge the roughly 74B/120B base weights.

Reload the composite model, then attach the adapter:

```python
import json
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

adapter = "outputs/bolt"
with open(f"{adapter}/stateset_training.json", encoding="utf-8") as stream:
    receipt = json.load(stream)
base = AutoModelForImageTextToText.from_pretrained(
    receipt["model_id"], revision=receipt["model_revision"],
    trust_remote_code=True, dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, adapter)
processor = AutoProcessor.from_pretrained(adapter, trust_remote_code=True)
model.eval()
```

Place the model and processor inputs on your inference devices before calling
`generate`. Engine-specific quantization, serving, and multimodal RL rollouts
remain separate workflows.

Official references: [Super VL BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16),
[Bolt BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16).

## Text and tool RL on River

The River backend routes the BF16 model names above, Bolt's `-24GB` name, and
Super VL's `-NVFP4` name through their own pinned BF16 tokenizer/chat template.
This closes the renderer gap in River SDK 0.11, which recognizes Lightning
directly. It does **not** add these models to River's hosted model pool. River
must advertise the exact requested model through your account capabilities.
The adapter also maps River 0.11's legacy image-placeholder metadata lookup to
Nemotron's real `<image>` token ID. The SDK performs that lookup even for
text-only structured input. Encoding, decoding, vocabulary, and rollout token
IDs are unchanged; the compatibility path is tested against SDK lowering.
The public catalog checked on October 15 listed Lightning and did not list
Bolt or Super VL; account-specific availability may differ.

```bash
pip install -e '.[river]'
# Set RIVER_API_KEY securely in the environment.
stateset-agents river-preflight --live \
  --base-model nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16

# Prepare a synthetic policy/task smoke experiment without paid calls.
python -m stateset_agents.verticals.river_refund \
  --base-model nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 \
  --benchmark refund-policy-v2 --seed 42 --steps 1 \
  --train-count 8 --validation-count 4 --test-count 4 --concurrency 4 \
  --rollout-token-budget 98304 --output outputs/river-bolt-smoke --dry-run
```

After successful authenticated access checks, omit `--dry-run` to execute using
the exact Bolt ID returned by the account. The durable token reservation budget
includes training, validation, test, and retries; it caps generated tokens, not
dollars or input tokens. The experiment checks committed optimizer activity and
retains hosted checkpoints, validation selection, and test traces. It is a
small mechanics run, not a model-quality study. These River renderers support
text and executed tools; image/video RL needs a separate multimodal renderer.

On October 15, the authenticated Super VL NVFP4 path completed a one-batch
synthetic refund-policy run: 32 training trajectories, one SDK-reported optimizer
update, a finite nonzero gradient norm, and a step-1 training checkpoint.
Validation reward tied the baseline, so selection retained step 0; the selected
baseline passed three of four test cases. This verifies hosted training
mechanics without establishing a quality gain or full-size local BF16 training.
The account did not advertise Bolt. Reports, checkpoint references, and the
initial SDK failure are retained in
[`benchmark_results/training_contracts/20261015/nemotron-vl`](../benchmark_results/training_contracts/20261015/nemotron-vl/).

See [River setup](GETTING_STARTED_RIVER.md) and
[River's authoritative account-access documentation](https://docs.river.ai/guides/models/).
