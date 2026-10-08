# StateSet Agents Examples

Welcome to the StateSet Agents examples directory! This collection demonstrates how to use the framework for training conversational AI agents with Group Relative Policy Optimization (GRPO).

## 📚 Documentation

- **[Advanced Training Examples](./ADVANCED_TRAINING_README.md)** ⭐ NEW!
  - Distributed Multi-GPU Training
  - Custom Reward Functions
  - Advanced Optimization Techniques

- **[API Examples](./API_EXAMPLES_README.md)**
  - API Client Usage
  - Interactive Chatbot
  - WebSocket Integration

## 🚀 Quick Start Examples

### Five-Minute Demo

The fastest path from `pip install stateset-agents` to a curated training set — offline, no GPU, no API key:

```bash
bash examples/five_minute_demo.sh
```

Writes three sample customer-support conversation logs (OpenAI chat-completions format), ingests them with `stateset-agents ingest`, grades + curates them with `stateset-agents improve run --reward customer_support`, and prints the graded report. See also the Colab version: [`notebooks/improve_your_agent_5min.ipynb`](../notebooks/improve_your_agent_5min.ipynb).

### Hello World

The fastest way to get started - runs instantly with no downloads:

```bash
python examples/hello_world.py
```

**Features:**
- No model downloads required (uses stub mode)
- Complete agent creation and conversation flow
- Reward computation demonstration
- Training loop overview

### Quick Start

A simple stub-backed onboarding example showing the first end-to-end
training flow:

```bash
python examples/quick_start.py
```

**Features:**
- No model downloads required (uses stub mode by default)
- Basic conversation handling
- Environment setup and training loop wiring
- Reward computation and post-training conversation smoke test
- Clear upgrade path to swap in a real checkpoint later

## 🎓 Training Examples

### Small Gemma models: E2B and E4B

The `gemma4-e2b` and `gemma4-e4b` presets target
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) and
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it).
They use the packaged `stateset_agents.training.gemma4_small_starter` helpers
for **text-only** GSPO training with LoRA. Install the Gemma 4 dependencies
(Transformers 5.5+ is required for this architecture):

```bash
pip install -e '.[gemma4]'

# Preview without downloading weights or requiring a GPU.
python examples/finetune_gspo.py --model gemma4-e2b --starter-profile memory --dry-run

# Save an editable training configuration.
python examples/finetune_gspo.py --model gemma4-e4b --starter-profile memory \
  --write-config gemma4-e4b.json

# Run on a CUDA GPU. The memory profile uses 4-bit QLoRA.
python examples/finetune_gspo.py --model gemma4-e4b --config gemma4-e4b.json --no-dry-run
```

The same starters also ship as installed commands, without an examples checkout:

```bash
stateset-agents gemma-4-e2b --starter-profile memory --json
stateset-agents gemma-4-e4b --starter-profile memory --write-config gemma4-e4b.json
stateset-agents gemma-4-e4b --config gemma4-e4b.json --no-dry-run
```

Their default output directories are `outputs/gemma4_e2b_gspo` and
`outputs/gemma4_e4b_gspo`, respectively. Pass `--output-dir` to name a run.

The balanced profile uses BF16 LoRA, batch size 1, rank 16, and 1024/512
prompt/completion token limits. The memory profile uses 4-bit NF4 QLoRA,
two generations per prompt, and 512/256 token limits. These are conservative
starting points, not benchmark-tuned settings. The example trains against
built-in task scenarios and rewards; use your own environment, reward function,
and held-out evaluation for a useful domain-specific model.

For supervised fine-tuning on your own conversations, use the existing SFT path:

```bash
python -m stateset_agents.training.sft \
  --dataset sft_train.jsonl --base-model google/gemma-4-E2B-it \
  --output-dir outputs/gemma4_e2b_sft --num-epochs 3 \
  --lora-r 16 --max-length 1024 --per-device-batch-size 1
```

Each JSONL row has the shape
`{"messages":[{"role":"user","content":"Where is my order?"},{"role":"assistant","content":"Please share your order number."}]}`.
The SFT command uses unquantized LoRA and saves an adapter; on a host without
CUDA it prints a plan instead of training. Gemma 2 2B
(`google/gemma-2-2b-it`), Gemma 3 1B (`google/gemma-3-1b-it`), and Gemma 3 4B
(`google/gemma-3-4b-it`) can also be selected through `--base-model`; obtain
Hugging Face access for gated checkpoints first. Older Gemma chat templates
may require putting instructions in the first user message instead of a
separate system message. The legacy `gemma3` preset still points to Gemma 2
9B for compatibility; it does not select a Gemma 3 checkpoint.

Google's E2B/E4B labels mean **effective** parameters: their model cards list
about 5.1B/8B parameters including embeddings. Budget memory for embeddings,
activations, rollouts, and any reference model as well as quantized weights;
these presets do not guarantee a particular GPU capacity. Images/audio are
outside this text-training workflow. Local tests cover configuration, model
loading, and a tiny random Gemma 4 LoRA update; full E2B/E4B training quality
and GPU memory have not been benchmarked here.

### More small models: Qwen, SmolLM, Phi, Ministral, and Liquid

Install the training dependencies with `pip install -e '.[small-models]'`
(or `pip install 'stateset-agents[small-models]'` for an installed release
containing these starters).

| Preset | Installed command | Checkpoint |
|---|---|---|
| `qwen3.5-2b` | `stateset-agents qwen3-5-2b` | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) |
| `qwen3.5-4b` | `stateset-agents qwen3-5-4b` | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) |
| `qwen3.5-9b` | `stateset-agents qwen3-5-9b` | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) |
| `smollm3-3b` | `stateset-agents smollm3-3b` | [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |
| `phi4-mini` | `stateset-agents phi-4-mini` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| `ministral3-3b` | `stateset-agents ministral-3-3b` | [`mistralai/Ministral-3-3B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) |
| `ministral3-8b` | `stateset-agents ministral-3-8b` | [`mistralai/Ministral-3-8B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) |
| `lfm2.5-2.6b` | `stateset-agents lfm2-5-2-6b` | [`LiquidAI/LFM2.5-2.6B`](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |

All commands default to previews. They accept `--starter-profile balanced|memory|quality`,
`--write-config`, `--config`, `--output-dir`, and `--no-dry-run`.
The example driver accepts the preset names above through `--model`.

```bash
stateset-agents qwen3-5-4b --starter-profile memory --json
stateset-agents phi-4-mini --starter-profile memory --write-config phi-mini.json
stateset-agents phi-4-mini --config phi-mini.json --no-dry-run
python examples/finetune_gspo.py --model smollm3-3b --starter-profile memory --dry-run
```

Each installed command has its own default output directory. Balanced uses
BF16 LoRA with rank 16 and batch size 1; memory uses NF4 QLoRA and two
generations. These are starting configurations, not measured GPU capacity or
quality guarantees. The starters use the framework's built-in task scenarios
and rewards; replace those with your own environment and held-out evaluation.

Qwen adapters include linear-attention projections; Phi adapters include fused
QKV and gate/up projections; Liquid adapters include convolution projections
and its MLP layers. Text-only training excludes vision/audio encoders even
when projection names overlap. The Ministral presets deliberately load the
official **BF16** instruction checkpoints, which can then be quantized for
QLoRA, rather than starting from the FP8 inference checkpoints.

LFM2.5 support is experimental. It uses the **custom LFM license**, whereas the
Qwen, SmolLM, and Ministral entries use Apache 2.0 and Phi uses MIT. LFM2.5
always generates reasoning: even the memory profile reserves 1024 completion
tokens, and longer tasks may need more. Qwen and SmolLM also support reasoning
modes; evaluate final-answer quality and truncation, not just aggregate reward.

For SFT on your own chat-format JSONL, pass any checkpoint above directly:

```bash
python -m stateset_agents.training.sft \
  --dataset sft_train.jsonl --base-model microsoft/Phi-4-mini-instruct \
  --output-dir outputs/phi_mini_sft --num-epochs 3 \
  --lora-r 16 --max-length 1024 --per-device-batch-size 1
```

This SFT command uses unquantized LoRA and prints a plan on hosts without CUDA.
The presets select native Transformers implementations and explicit adapter
targets. SFT preserves the chat template's special tokens and trains on real
end-of-sequence tokens, including when EOS is also used for padding. Tests use
tiny random models to exercise real GSPO and SFT updates, adapter save/reload,
and agent generation for each architecture. Full pretrained-model training, quantized
GPU runs, and downstream task quality still need validation on suitable hardware.

### Additional small models and FunctionGemma

Install `pip install -e ".[small-models]"` from the repository, then use these
presets with the same balanced, memory (NF4 QLoRA), and quality profiles:

| Preset | Model card | CLI command |
|---|---|---|
| `granite4-micro` | [`ibm-granite/granite-4.0-micro`](https://huggingface.co/ibm-granite/granite-4.0-micro) | `stateset-agents granite-4-micro` |
| `llama3.2-1b` | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | `stateset-agents llama-3-2-1b` |
| `llama3.2-3b` | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | `stateset-agents llama-3-2-3b` |
| `qwen3-4b-instruct` | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `stateset-agents qwen3-4b-instruct` |
| `olmo3-7b` | [`allenai/Olmo-3-7B-Instruct`](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | `stateset-agents olmo-3-7b` |
| `deepseek-r1-1.5b` | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | `stateset-agents deepseek-r1-1-5b` |
| `deepseek-r1-7b` | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | `stateset-agents deepseek-r1-7b` |

```bash
stateset-agents granite-4-micro --starter-profile memory --json
stateset-agents granite-4-micro --starter-profile memory --write-config granite.json
stateset-agents granite-4-micro --config granite.json --no-dry-run
```

Granite Micro targets both attention and its fused `input_linear` /
`output_linear` MLP modules. The framework disables generation caching for the
attention-only Granite architecture to avoid a Transformers 5.14 recurrent-cache
error; this costs decoding speed. Hybrid Granite variants are unaffected.
DeepSeek R1 distills reserve 2048 completion tokens even in the memory profile
(4096 in quality); increase this if reasoning truncates before the final answer.
Llama 3.2 requires access to the gated checkpoint and its community license.
Granite, Qwen3 Instruct, and Olmo use Apache 2.0; the listed DeepSeek distills use MIT.

All seven checkpoints can also use the SFT command above with their model ID.
Tiny random architecture tests cover real SFT/GSPO updates, adapter reload, and
agent generation. Full checkpoint quality and GPU quantization remain unverified.

[FunctionGemma 270M](https://huggingface.co/google/functiongemma-270m-it) is
registered as `functiongemma-270m` for **SFT only**. It uses the Gemma license
and requires Hugging Face access. Its tool-call format needs developer messages
and per-row `tools` schemas. Use structured assistant `tool_calls` with
argument objects; the model's own chat template renders them. SFT preserves
those objects before constructing the tokenized dataset, including different
schemas and argument types across rows.

The [commerce JSONL example](data/functiongemma_commerce.jsonl) contains two
illustrative tool-selection demonstrations; expand it with task-specific
training examples and separate evaluation data before a real run.

```bash
python -m stateset_agents.training.sft \
  --dataset examples/data/functiongemma_commerce.jsonl \
  --base-model google/functiongemma-270m-it \
  --output-dir outputs/functiongemma_commerce \
  --num-epochs 3 --lora-r 16 --max-length 1024 --per-device-batch-size 1
```

On CPU this command prints a plan. The generic conversational GSPO driver
rejects FunctionGemma with an SFT command hint: it does not yet have a dedicated
tool-execution environment/reward integration. FunctionGemma's architecture and
schema preservation are tested locally; the gated vendor tokenizer and full
checkpoint have not been validated. The SFT `--eval-prompts` helper takes plain
text prompts and is not a tool-calling evaluation harness.

### River: executed commerce outcomes

[`river_refund_rl.py`](river_refund_rl.py) trains a remote agent against an isolated
refund ledger, selects checkpoints on validation, and records per-case held-out
test evidence. Prepare cases offline with
`python examples/river_refund_rl.py --dry-run --output outputs/refund-42`.
Training and evaluation-only runs require the River extra and credentials.
The same runner ships in wheels as `python -m stateset_agents.training.river_refund`.
Use `benchmark plan-agent-study` to freeze a six-seed, four-arm campaign and
`benchmark audit-agent-study --strict` to check its provenance and learning gates.
Compare matched runs using `stateset-agents benchmark compare-agents`; see the
[River provider reference](../docs/RIVER_PROVIDER.md) for the full workflow.
Add `--benchmark refund-policy-v2` for policy reasoning over partial refunds,
return-window boundaries, chargebacks, and misleading customer notes, with
per-family evaluation and regression gates.
Use `stateset-agents benchmark prepare-refund-data` for verified reference SFT
demonstrations, or run the example with `--collect-only` and pass its candidates
to `stateset-agents benchmark filter-refund-data` for rejection-sampling SFT.
The exporter and filter run offline; collecting model candidates uses River.

### Basic Training

#### 1. Complete GRPO Training

Full-featured GRPO training example with all components:

```bash
python examples/complete_grpo_training.py
```

**Covers:**
- Agent initialization
- Environment creation
- Reward function setup
- Complete training loop
- Checkpoint saving

#### 2. TRL Integration

Using Hugging Face TRL library for GRPO:

```bash
python examples/train_with_trl_grpo.py
```

**Features:**
- TRL GRPO trainer integration
- Hugging Face model support
- Dataset handling
- Automatic logging

#### 3. Train Reward Models

Learn to train custom reward models:

```bash
python examples/train_reward_model.py
```

**Covers:**
- Neural reward model training
- Reward dataset preparation
- Model evaluation
- Integration with GRPO

#### 4. Symbolic Physics Discovery

Train on toy symbolic constraints with hidden targets in metadata:

```bash
python examples/physics_symbolic_discovery.py --tasks examples/data/symbolic_physics_tasks.jsonl
```

**Covers:**
- Task schema with constraints + derived variables
- Constraint-based symbolic rewards
- Metadata-aware GSPO queries

Evaluate model outputs against constraints:

```bash
python examples/physics_symbolic_evaluate.py \
    --tasks examples/data/symbolic_physics_tasks.jsonl \
    --predictions /path/to/predictions.jsonl
```

### Advanced Training ⭐ NEW!

See **[Advanced Training README](./ADVANCED_TRAINING_README.md)** for detailed guides on:

#### Distributed Multi-GPU Training

Scale training across multiple GPUs:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    examples/distributed_multi_gpu_training.py \
    --model gpt2 \
    --task customer_service
```

#### Custom Reward Functions

Create domain-specific rewards:

```bash
# Test custom rewards
python examples/custom_reward_functions.py

# View available rewards
python examples/custom_reward_functions.py --list-rewards
```

#### Advanced Optimization

Optimize training with cutting-edge techniques:

```bash
python examples/advanced_optimization_techniques.py \
    --model gpt2 \
    --mixed-precision bf16 \
    --compile
```

## 🎯 Domain-Specific Examples

### Customer Service

#### Production Ready Customer Service

Enterprise-grade implementation:

```bash
python examples/production_ready_customer_service.py
```

**Includes:**
- Error handling and resilience
- Monitoring and logging
- Performance optimization
- Deployment-ready code

### Technical Support

RAG-enabled technical support agent:

```bash
python examples/rag_agent_example.py
```

**Features:**
- Retrieval-Augmented Generation
- Document search
- Technical knowledge base
- Code analysis capabilities

## 🔧 Fine-Tuning Examples

### GSPO (Group Sequence Policy Optimization)

StateSet Agents includes GSPO, a more stable alternative to GRPO.

#### Unified Finetune Driver

`examples/finetune_gspo.py` (backed by `examples/model_presets.py`) is a
single parameterized driver covering the common agent/reward/GSPO-config
wiring shared by every per-model script below:

```bash
python examples/finetune_gspo.py --list-models
python examples/finetune_gspo.py --model kimi-k3 --dry-run
python examples/finetune_gspo.py --model glm5.1 --task customer_service --no-dry-run
```

Use it for a quick preview of any supported preset (`muse-glimmer`, `nemotron-3-5`, `qwen3.8-27b`, `qwen3.8-flash-next`, `qwen3-coder`, `gpt-oss`, `deepseek-v4`, `kimi-k3`, `kimi-k2.5`,
`kimi-k2.6`, `glm5.1`, `glm5.2`, `glm5.3-flash`, `qwen3`, `qwen3.5-0.8b`, `qwen3.5-27b`,
`gemma3`, `gemma4-31b`, `llama3`, `mistral`), or a full real run. `--dry-run`
defaults to `True`; pass `--no-dry-run` to actually invoke the training
entry point (the packaged starter's `run_<name>_config` for starter-backed
presets, or `stateset_agents.training.gspo_entrypoints.train_with_gspo`
otherwise).

The driver also absorbs every flag family shared across the packaged-starter
scripts: `--use-lora/--no-lora`, `--use-4bit/--use-8bit`, `--use-vllm`,
`--wandb`/`--wandb-project` (wired into `GSPOConfig.report_to`/
`wandb_project`/`wandb_tags` on both the starter and non-starter paths),
`--export-merged` (wired into `export_merged_model_for_serving` for
non-starter presets; exits with a clear error for starter-backed presets,
since none of the packaged starters currently support merge export),
`--learning-rate`, `--epochs`/`--steps`, `--iterations` (maps to a starter's
`num_outer_iterations` override; errors clearly for non-starter presets
instead of being silently dropped), and, for presets whose
`ModelPreset.starter_module` is set, `--starter-profile
{balanced,memory,quality}`, `--config PATH`, `--write-config PATH`, and
`--list-profiles`. Four per-model scripts whose entire CLI is now
reproducible this way —
`finetune_kimi_k3_gspo.py`, `finetune_kimi_k2_6_gspo.py`,
`finetune_gemma4_31b_gspo.py`, and `finetune_qwen3_5_0_8b_gspo.py` — are now
thin deprecated forwarders onto `examples/finetune_gspo.py --model <preset>`
and will be removed in a future release. `finetune_muse_glimmer_gspo.py` is a
thin forwarder of the same shape for the `muse-glimmer` preset
(`meta-models/Muse-Glimmer-30B`); prefer
`python examples/finetune_gspo.py --model muse-glimmer` directly.
`finetune_nemotron_3_5_gspo.py` is a thin forwarder of the same shape for the
`nemotron-3-5` preset (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`);
prefer `python examples/finetune_gspo.py --model nemotron-3-5` directly.
`finetune_qwen3_8_27b_gspo.py` is a thin forwarder of the same shape for the
`qwen3.8-27b` preset (`Qwen/Qwen3.8-27B`); prefer
`python examples/finetune_gspo.py --model qwen3.8-27b` directly.
`finetune_qwen3_coder_gspo.py` is a thin forwarder of the same shape for the
`qwen3-coder` preset (`Qwen/Qwen3-Coder-30B-A3B-Instruct`); prefer
`python examples/finetune_gspo.py --model qwen3-coder` directly.
`finetune_gpt_oss_gspo.py` is a thin forwarder of the same shape for the
`gpt-oss` preset (`openai/gpt-oss-20b`); prefer
`python examples/finetune_gspo.py --model gpt-oss` directly.
`finetune_deepseek_v4_gspo.py` is a thin forwarder of the same shape for the
`deepseek-v4` preset (`deepseek-ai/DeepSeek-V4-Flash`); prefer
`python examples/finetune_gspo.py --model deepseek-v4` directly. The other dedicated per-model
scripts below are kept because they carry genuinely unique logic the driver
does not (and, for some, should not) generalize: `finetune_glm5_1_gspo.py`
and `finetune_glm5_2_gspo.py` add serving-only flags (`--fp8-serving`,
`--disable-auto-tool-choice`); `finetune_kimi_k25_gspo.py`,
`finetune_qwen3_gspo.py`, `finetune_qwen3_5_27b_gspo.py`,
`finetune_gemma3_gspo.py`, `finetune_llama3_gspo.py`, and
`finetune_mistral_gspo.py` branch internally across multiple model sizes /
MoE variants (a single `ModelPreset` only captures one representative
branch each — see each preset's `notes` field in `examples/model_presets.py`
for which branch); `finetune_kimi_k2_5_gspo.py` is already a deprecated
forwarder onto `finetune_kimi_k25_gspo.py`.

#### Qwen Models

Fine-tune Qwen models with GSPO:

```bash
python examples/finetune_qwen3_5_0_8b_gspo.py --task customer_service
python examples/finetune_qwen3_5_0_8b_gspo.py --starter-profile memory --dry-run
python examples/finetune_qwen3_5_0_8b_gspo.py --list-profiles
python examples/finetune_qwen3_5_27b_gspo.py --dry-run
python examples/finetune_qwen3_5_27b_gspo.py --task customer_service --output-dir /models/qwen3-5-27b
```

See [QWEN3_FINETUNING_GUIDE.md](../docs/QWEN3_FINETUNING_GUIDE.md) for a getting-started walkthrough for post-training `Qwen/Qwen3.5-0.8B`, including the built-in `balanced`, `memory`, and `quality` starter profiles and the new profile-discovery mode. The family-wide fallback script remains `examples/finetune_qwen3_gspo.py`. `examples/finetune_qwen3_5_0_8b_gspo.py` is now a deprecated forwarder; prefer `python examples/finetune_gspo.py --model qwen3.5-0.8b`.
For `Qwen/Qwen3.5-27B`, the dedicated starter emits `serving_manifest.json`
plus merged checkpoints so you can render Helm values or deploy the raw
Kubernetes manifests in `deployment/kubernetes/`.

#### Kimi Models

Fine-tune Moonshot Kimi models with GSPO:

```bash
python examples/finetune_kimi_k2_6_gspo.py --dry-run
python examples/finetune_kimi_k2_6_gspo.py --starter-profile memory --dry-run
python examples/finetune_kimi_k2_6_gspo.py --list-profiles
python examples/finetune_kimi_k3_gspo.py --dry-run
python examples/finetune_kimi_k3_gspo.py --list-profiles
python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task customer_service
```

`examples/finetune_kimi_k2_6_gspo.py` and `examples/finetune_kimi_k3_gspo.py` are now deprecated forwarders onto `examples/finetune_gspo.py --model kimi-k2.6` / `--model kimi-k3` (the same `balanced`/`memory`/`quality` starter profile flow, including `--config`/`--write-config`/`--list-profiles`, is reproduced by the driver). `examples/finetune_kimi_k3_gspo.py` covers the provisional `moonshotai/Kimi-K3` ID (HF weights pending as of 2026-07-16). `examples/finetune_kimi_k2_5_gspo.py` is a deprecated forwarder that now delegates to `examples/finetune_kimi_k25_gspo.py` (a strict superset of its flags) and will be removed in a future release. `examples/kimi_k25_rewards.py` and `examples/kimi_k25_config.py` provide Kimi-K2.5-specific reward functions and hyperparameter defaults used by the finetune script; see `examples/kimi_k25/README.md` for the full walkthrough and `examples/kimi_k25/live_smoke_checks.py` for a live (network-dependent) model-loading smoke check.

#### Gemma Models

Fine-tune Google Gemma models:

```bash
python examples/finetune_gemma4_31b_gspo.py --dry-run
python examples/finetune_gemma4_31b_gspo.py --starter-profile memory --dry-run
python examples/finetune_gemma4_31b_gspo.py --no-dry-run --task customer_service
```

The dedicated Gemma 4 starter targets `google/gemma-4-31B-it` with GSPO-ready
QLoRA defaults for StateSet Agents. The older family-wide fallback script remains
`examples/finetune_gemma3_gspo.py` for Gemma 2 era checkpoints.
`examples/finetune_gemma4_31b_gspo.py` is now a deprecated forwarder; prefer
`python examples/finetune_gspo.py --model gemma4-31b`.

#### GLM Models

Fine-tune Zhipu AI's GLM 5.1 (754B MoE):

```bash
python examples/finetune_glm5_1_gspo.py --dry-run
python examples/finetune_glm5_1_gspo.py --starter-profile memory --dry-run
python examples/finetune_glm5_1_gspo.py --model your-org/GLM-5.1-FP8 --fp8-serving --dry-run
python examples/finetune_glm5_1_gspo.py --no-dry-run --task customer_service --output-dir /models/glm5-1
```

The dedicated GLM 5.1 starter targets `zai-org/GLM-5.1` (BF16) and a private
alias such as `your-org/GLM-5.1-FP8` for single-host serving. See
[GLM5_1_HOSTING_PLAN.md](../docs/GLM5_1_HOSTING_PLAN.md) for the full
deployment recipe (Helm values, K8s manifests, multi-node topology, and
the Helm values renderer in `scripts/render_glm5_1_helm_values.py`).
`examples/finetune_glm5_2_gspo.py` is the equivalent starter for
`zai-org/GLM-5.2`, mirroring the GLM 5.1 flags and profiles.

`examples/gemma4_config.py`, `examples/glm5_1_config.py`,
`examples/glm5_2_config.py`, `examples/kimi_k2_6_config.py`,
`examples/kimi_k3_config.py`, and `examples/qwen3_5_config.py` are
backward-compatible re-export shims over the packaged starter modules in
`stateset_agents.training.*_starter`; import from the starter module
directly in new code.

#### Llama Models

Fine-tune Meta Llama models:

```bash
python examples/finetune_llama3_gspo.py \
    --model meta-llama/Llama-3.2-3B \
    --task customer_service \
    --use-lora --use-4bit
```

#### Mistral Models

Fine-tune Mistral models:

```bash
python examples/finetune_mistral_gspo.py \
    --model mistralai/Mistral-7B-v0.1 \
    --task customer_service \
    --use-lora
```

#### Code Assistant

Fine-tune for code generation:

```bash
python examples/finetune_code_assistant.py
```

**Features:**
- Code-specific reward functions
- Syntax validation
- Multi-language support

## 🎨 Framework Showcases

### GRPO Showcase

Comprehensive demonstration of GRPO capabilities:

```bash
python examples/grpo_showcase.py
```

**Demonstrates:**
- Multi-turn trajectory generation
- Group advantage computation
- Policy gradient updates
- Value function training
- Reward shaping

## 🔌 API Examples

See **[API Examples README](./API_EXAMPLES_README.md)** for complete API documentation.

### API Client (Async)

Full-featured async client:

```bash
python examples/api_client_example.py
```

### API Client (Simple)

Synchronous client for quick integration:

```bash
python examples/api_client_simple.py
```

### Interactive Chatbot

CLI chatbot using the API:

```bash
python examples/interactive_chatbot.py
```

## 🧪 Experimental Examples

### HPO (Hyperparameter Optimization)

Automatic hyperparameter tuning:

```bash
python examples/hpo_training_example.py
```

**Features:**
- Optuna integration
- Automatic search space
- Multi-objective optimization
- Best config selection

### Backend Switching

Dynamic backend switching:

```bash
python examples/backend_switch_demo.py
```

**Demonstrates:**
- Stub backend for testing
- Real model backend
- Runtime switching

## 🧩 Additional Examples

Miscellaneous standalone scripts not covered above:

- `examples/nsr_verified_reward.py` — the NSR neuro-symbolic verifier reward end to end: builds a decision request with rules/facts, scores a trajectory with `NSRVerifierReward`, and reports the outcome. Runs offline against an injected fake verifier by default:

```bash
python examples/nsr_verified_reward.py            # offline, no server or keys
python examples/nsr_verified_reward.py --live     # against a real NSR API
```

- `examples/model_presets.py` — the model-preset registry (`PRESETS`) backing `examples/finetune_gspo.py`; import `get_preset`/`list_preset_names` to reuse the hyperparameters in your own scripts.
- `examples/advanced_features_demo.py` — end-to-end demo of curriculum learning, multi-agent coordination, offline RL (CQL/IQL), Bayesian uncertainty, and few-shot adaptation.
- `examples/auto_research.py` — the autonomous hyperparameter-optimization research loop.
- `examples/auto_research_quickstart.py` — a minimal, copy-and-modify runnable version of the auto-research loop.
- `examples/continual_learning_planning.py` — long-term planning plus continual learning (replay + LwF) across multiple tasks.
- `examples/customer_service_agent.py` — a focused GRPO customer-service training example (see "I want to... train a customer service agent" above).
- `examples/custom_math_env.py` — a custom `MathEnvironment` showing how to build your own environment and reward.
- `examples/qwen3b_gspo_demo.py` — a minimal single-GPU GSPO fine-tune of `Qwen/Qwen2.5-3B` with LoRA + 8-bit loading.
- `examples/simple_rl_training.py` — the smallest possible GSPO training loop (GPT-2 + a toy environment).
- `examples/training_modes_quickstart.py` — stub-backed tour of all 5 unified `train()` modes (online, offline, RLAIF, etc.).
- `examples/train_with_gspo.py` — GSPO training walkthrough with `--task`/`--model`/`--use-gspo-token` CLI flags.
- `examples/trl_grpo_demo.py` — the simplest TRL GRPO integration demo.

## 📦 Prerequisites

### Basic Requirements

```bash
pip install stateset-agents
```

### Development Requirements

For all examples:

```bash
pip install stateset-agents[dev]
```

### Optional Dependencies

For TRL integration:
```bash
pip install stateset-agents[trl]
```

For API examples:
```bash
pip install stateset-agents[api]
```

For HPO:
```bash
pip install stateset-agents[hpo]
```

For all features:
```bash
pip install stateset-agents[dev,api,trl,hpo]
```

## 🎯 Examples by Use Case

### I want to...

#### ...get started quickly
→ `hello_world.py` or `quick_start.py`

#### ...train a customer service agent
→ `production_ready_customer_service.py` or `customer_service_agent.py`

#### ...train on multiple GPUs
→ `distributed_multi_gpu_training.py` ⭐ NEW!

#### ...create custom reward functions
→ `custom_reward_functions.py` ⭐ NEW!

#### ...optimize training performance
→ `advanced_optimization_techniques.py` ⭐ NEW!

#### ...fine-tune a specific model
→ Choose from:
- `finetune_qwen3_5_0_8b_gspo.py`
- `finetune_qwen3_5_27b_gspo.py`
- `finetune_qwen3_gspo.py`
- `finetune_gemma3_gspo.py`
- `finetune_gemma4_31b_gspo.py`
- `finetune_glm5_1_gspo.py`
- `finetune_llama3_gspo.py`
- `finetune_mistral_gspo.py`

#### ...integrate with my application
→ `api_client_example.py` or `api_client_simple.py`

#### ...build a chatbot
→ `interactive_chatbot.py`

#### ...understand GRPO internals
→ `grpo_showcase.py`

#### ...find optimal hyperparameters
→ `hpo_training_example.py`

## 📝 Example Structure

Each example follows this structure:

```python
"""
Example Title

Description of what the example demonstrates.

Requirements:
    - List of dependencies

Usage:
    # How to run the example
    python examples/example_name.py [options]
"""

# Imports
import asyncio
from stateset_agents import MultiTurnAgent
# ...

# Configuration
# ...

# Main functionality
async def main():
    # Example code
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛠️ Running Examples

### Basic Execution

```bash
python examples/<example_name>.py
```

### With Arguments

```bash
python examples/<example_name>.py --help  # See all options
python examples/<example_name>.py --option value
```

### From Python

```python
import asyncio
from examples import example_module

asyncio.run(example_module.main())
```

## 📚 Learning Path

### Beginner

1. `hello_world.py` - Understand basic concepts
2. `quick_start.py` - Run your first end-to-end training example
3. `api_client_simple.py` - Integrate with applications

### Intermediate

4. `customer_service_agent.py` - Build domain-specific agents
5. `custom_reward_functions.py` - Create custom rewards ⭐
6. `train_with_trl_grpo.py` - Use HuggingFace TRL

### Advanced

7. `distributed_multi_gpu_training.py` - Scale to multiple GPUs ⭐
8. `advanced_optimization_techniques.py` - Optimize performance ⭐
9. `finetune_qwen3_5_0_8b_gspo.py` - Run the Qwen3.5-0.8B starter path
10. `finetune_qwen3_5_27b_gspo.py` - Run the Qwen3.5-27B starter path for k8s/vLLM serving
11. `finetune_qwen3_gspo.py` - Fine-tune broader Qwen model variants
12. `finetune_gemma4_31b_gspo.py` - Run the Gemma 4 31B starter path
13. `finetune_glm5_1_gspo.py` - Run the GLM 5.1 (754B MoE) starter path for multi-node vLLM serving
14. `hpo_training_example.py` - Automated hyperparameter tuning

## 🐛 Troubleshooting

### Common Issues

#### Import Errors

```bash
pip install -e ".[dev]"  # Install from source
```

#### CUDA Out of Memory

See [Advanced Training README](./ADVANCED_TRAINING_README.md#troubleshooting) for memory optimization tips.

#### Slow Training

Check [Performance Tips](./ADVANCED_TRAINING_README.md#performance-tips) for optimization strategies.

## 📞 Support

- **Documentation**: https://stateset-agents.readthedocs.io/
- **Discord**: https://discord.gg/stateset
- **Issues**: https://github.com/stateset/stateset-agents/issues

## 🤝 Contributing

Want to add an example? See [CONTRIBUTING.md](../CONTRIBUTING.md)

## 📄 License

See [LICENSE](../LICENSE) for details.

---

**Made with ❤️ by the StateSet Team**
