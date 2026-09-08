# Supported Models

`stateset-agents` fine-tunes any HuggingFace causal-LM checkpoint that loads via
`AutoModelForCausalLM` / `AutoTokenizer`. Some model families ship as **first-class
starters** with packaged configs, CLI entrypoints, Kubernetes training manifests, and
profile presets; others are referenced in examples, hosting plans, or finetuning
guides. The rest ("effectively supported") work through the generic training flow.

## First-class starters

Each first-class starter provides:

- Dedicated module in `stateset_agents/training/<family>_starter.py`
- Three profiles: `balanced` (default), `memory` (low-VRAM QLoRA), `quality` (larger context/rollouts)
- Four task presets: `customer_service`, `technical_support`, `sales`, `conversational`
- Packaged example script under `examples/finetune_*_gspo.py`

| Model | HuggingFace ID | CLI command | Example script | Notes |
|---|---|---|---|---|
| **Qwen 3.5 0.8B** | `Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-0.8B` | `stateset-agents qwen3-5-0-8b` | `examples/finetune_qwen3_5_0_8b_gspo.py` | Smallest first-run target; good for CI smoke tests |
| **Gemma 4 31B IT** | `google/gemma-4-31B-it` | `stateset-agents gemma-4-31b` | `examples/finetune_gemma4_31b_gspo.py` | QLoRA 4-bit by default |
| **Kimi-K2.6** | `moonshotai/Kimi-K2.6` | `stateset-agents kimi-k2-6` | `examples/finetune_kimi_k2_6_gspo.py` | GKE Autopilot + Standard manifests shipped |
| **GLM 5.1** | `zai-org/GLM-5.1` (754B MoE) | — (module + example only) | `examples/finetune_glm5_1_gspo.py` | QLoRA-only; vLLM generation; FP8 alias supported |
| **GLM 5.2** | `zai-org/GLM-5.2` (754B MoE) | — (module + example only) | `examples/finetune_glm5_2_gspo.py` | QLoRA-only; vLLM generation; FP8 alias supported |
| **Kimi-K3** | `moonshotai/Kimi-K3` *(provisional — HF weights pending as of 2026-07-16)* | `stateset-agents kimi-k3` | `examples/finetune_kimi_k3_gspo.py` | Mirrors K2.6 presets pending official specs |
| **Muse Glimmer 30B** | `meta-models/Muse-Glimmer-30B` | `stateset-agents muse-glimmer` | `examples/finetune_muse_glimmer_gspo.py` | Meta's open agentic model (Aug 2026); dense 30B, 131K ctx, Apache-2.0; QLoRA 4-bit by default |
| **Qwen3.8 27B** | `Qwen/Qwen3.8-27B` (also `Qwen/Qwen3.8-27B-FP8`, inference-oriented) | `stateset-agents qwen3-8-27b` | `examples/finetune_qwen3_8_27b_gspo.py` | Alibaba's multimodal hybrid-attention model (2026-08-05, Apache-2.0); 27.8B params (~56GB BF16 — budget 160GB disk and an 80GB card or `--gpu-count 2`), 64 text layers, 256K ctx; LoRA targets standard attention + Mamba-style `linear_attn` + MLP (verified against the weight map); vision tower excluded |
| **Qwen3 Coder 30B A3B** | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `stateset-agents qwen3-coder` | `examples/finetune_qwen3_coder_gspo.py` | Alibaba's open coding MoE (30B total / ~3B active, 128 experts / 8 active, 256K ctx, Apache-2.0); attention-only LoRA (128-expert MoE MLPs are impractical LoRA targets); FP8 variant is inference-oriented |
| **gpt-oss 20B** | `openai/gpt-oss-20b` (also `openai/gpt-oss-120b`, multi-GPU) | `stateset-agents gpt-oss` | `examples/finetune_gpt_oss_gspo.py` | OpenAI's open-weight reasoning MoE (32 experts / 4 active, 131K ctx, Apache-2.0; adjustable reasoning effort, harmony format); attention-only LoRA verified against the weight map |
| **DeepSeek V4 Flash** | `deepseek-ai/DeepSeek-V4-Flash` (also `...-Flash-Base`; NVFP4/FP8 repos are inference-only) | `stateset-agents deepseek-v4` | `examples/finetune_deepseek_v4_gspo.py` | Large MoE with MLA (256 routed experts / 6 active, up to 1M positions, MIT); QLoRA-only, vLLM generation; LoRA targets the MLA projections `wq_a`/`wq_b`/`wkv`/`wo_a`/`wo_b` (verified against the weight map) |
| **Nemotron 3.5 Lightning 30B A3B** | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `stateset-agents nemotron-3-5` | `examples/finetune_nemotron_3_5_gspo.py` | NVIDIA's hybrid Mamba-2/attention/MoE model (Aug 2026); 30B total / ~3B active, 256K ctx, OpenMDW-1.1; QLoRA 4-bit on the BF16 checkpoint (NVFP4 is inference-only) |

Each CLI command supports `--list-profiles`, `--write-config`, `--config`,
`--starter-profile {balanced,memory,quality}`, `--task`, `--json-output`, and
`--dry-run` / `--no-dry-run`.

## Reference models (examples, hosting plans, finetuning guides)

These have example scripts, Kubernetes manifests, or dedicated docs but are not
exposed as `*_starter.py` modules with profile presets.

| Model | HuggingFace ID | Surface |
|---|---|---|
| **GLM 5.3 Flash** | `zai-org/GLM-5.3-Flash` (320B MoE / 18B active) | `python examples/finetune_gspo.py --model glm5.3-flash --dry-run`; install `stateset-agents[glm53]`. Native multimodal `glm5_next`, FP8, 1M context; hybrid linear/sparse attention LoRA targets verified from the official weight index. StateSet text-only RL loads the composite model automatically; use `AutoProcessor`, vLLM, or SGLang for image/video inference. See [GLM5_3_FLASH.md](GLM5_3_FLASH.md). |
| **Qwen3.8 Flash Next** | `Qwen/Qwen3.8-Flash-Next` (125B MoE / 6B active + 51B n-gram embeddings) | `python examples/finetune_gspo.py --model qwen3.8-flash-next --dry-run`; install `stateset-agents[qwen38next]`. Native multimodal `qwen4_exp`, 262K native context; Gated DeltaNet/Qwen Sparse Attention LoRA targets verified from the official weight index. Text-only RL uses the composite loader; use `AutoProcessor`, vLLM, SGLang, or TokenSpeed for multimodal inference. See [QWEN3_8_FLASH_NEXT.md](QWEN3_8_FLASH_NEXT.md). |
| **Inkling-Small** | `thinkingmachines/Inkling-Small` | First-class Tinker remote-autograd SFT via `train-remote --provider tinker`. The result is a hosted sampler/state pointer rather than local Hugging Face weights. Integration is unit-tested; live certification is still pending. See [MANAGED_TRAINING_PROVIDERS.md](MANAGED_TRAINING_PROVIDERS.md). |
| **Qwen 3.5 27B** | `Qwen/Qwen3.5-27B` | `examples/finetune_qwen3_5_27b_gspo.py`, `deployment/kubernetes/qwen3-5-27b-training-job.yaml`, `docs/QWEN3_5_27B_MINIMAL_HOSTING_PLAN.md` |
| **Qwen 3** (general) | Qwen 3 family | `examples/finetune_qwen3_gspo.py`, `docs/QWEN3_FINETUNING_GUIDE.md` |
| **Qwen 2.5 3B Instruct** | `Qwen/Qwen2.5-3B-Instruct` | Offline GRPO example in `training/offline_grpo_trainer.py` |
| **Kimi-K2.5** | `moonshotai/Kimi-K2.5` | `examples/finetune_kimi_k2_5_gspo.py`, `deployment/kubernetes/kimi-k25-*.yaml`, `docs/KIMI_K25_GKE_AUTOPILOT.md`, `docs/KIMI_K25_GKE_STANDARD.md` |
| **Gemma 3 / Gemma 2 27B IT** | `google/gemma-2-27b-it` | `examples/finetune_gemma3_gspo.py`, `docs/GEMMA3_FINETUNING_GUIDE.md` |
| **Llama 3** | Llama 3 family | `examples/finetune_llama3_gspo.py` |
| **Llama 2 7B** | `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-7b-chat-hf` | vLLM backend default (`training/vllm_backend.py`), QUICKSTART examples |
| **Mistral 7B** | `mistralai/Mistral-7B-v0.1`, `-Instruct-v0.1` | `examples/finetune_mistral_gspo.py`, default RLAIF reward model |
| **GPT-2** | `gpt2` | Smoke-test default across README/QUICKSTART examples |

## Effectively supported (generic flow)

Because the training stack is `AutoModelForCausalLM` + `AutoTokenizer` + optional
PEFT/QLoRA + vLLM, any HuggingFace causal-LM compatible with TRL GRPO should
train end-to-end. You pass the HF identifier to `AgentConfig(model_name=...)`:

```python
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

agent = MultiTurnAgent(AgentConfig(model_name="<huggingface-id>"))
```

Examples that work with the generic flow (may need LoRA target-module tuning):

- Llama 3.x variants (3B / 8B / 70B)
- DeepSeek-V2 / V3 / Coder
- Phi-3 Mini / Medium
- Yi-1.5
- Mixtral 8x7B / 8x22B
- Qwen 2.5 / Qwen 3 / Qwen 3.5 size variants not listed above
- Any other causal LM with a `chat_template` on its tokenizer

If your checkpoint needs non-standard attention, quantization, or custom LoRA
target modules, start from the closest first-class starter and adapt the
`*_LORA_TARGET_MODULES` list in that starter module.

## Algorithm compatibility

All supported models train with any of the algorithms shipped in
`stateset_agents/training/`:

| Algorithm | Trainer module |
|---|---|
| GRPO | `trl_grpo_trainer.py` |
| GSPO | `gspo_trainer.py`, `gspo_token_trainer.py` |
| GEPO | `gepo_trainer.py` |
| DAPO | `dapo_trainer.py` |
| VAPO | `vapo_trainer.py` |
| PPO | `base_trainer.py` (PPO path) |
| RLAIF | `rlaif_trainer.py` |
| Offline GRPO | `offline_grpo_trainer.py` |
| Decision Transformer | `decision_transformer.py` |

Offline RL variants (BCQ, BEAR, CQL, IQL) are available via the offline
trainer; see `docs/OFFLINE_RL_GUIDE.md` if present.

## Stub mode for tests

For CI, smoke tests, or local dev without downloading weights, use the stub
backend:

```python
AgentConfig(model_name="stub://my-test", use_stub_model=True)
```

See `TESTING.md` for the full stub-backend fixture catalog.

## Adding a new first-class starter

Every `stateset_agents/training/*_starter.py` module is built from one
`StarterSpec` by `stateset_agents/training/starter_factory.py`. A family
module contains exactly three things:

1. its constants (`<FAMILY>_BASE_MODEL`, optional `<FAMILY>_POST_TRAINED_MODEL`,
   `<FAMILY>_TASK_CHOICES`, `<FAMILY>_STARTER_PROFILE_CHOICES`) and any extra
   family-only constants or helpers (for example GLM's serving
   recommendations);
2. a hand-written `validate_<family>_config(config) -> list[str]` holding the
   family's warning rules (learning-rate bounds, batch limits, model-name
   checks, required quantization, and so on);
3. a `SPEC = StarterSpec(...)` carrying the data that varies between families
   (labels, symbol prefix and function infix, base model and variants, LoRA
   targets, profile descriptions and overrides, the system-prompt intro, the
   config defaults that differ from the shared ones, optional extra config
   fields such as `use_vllm`, W&B tags, and agent-config kwargs), followed by
   `globals().update(build_starter(SPEC, logger))` and
   `__all__ = starter_all(_SYMBOLS)`.

`build_starter` generates the public contract every family shares: the
constants above plus `<FAMILY>_SUPPORTED_VARIANTS`,
`<FAMILY>_STARTER_PROFILE_DESCRIPTIONS`, `<FAMILY>_DEFAULT_OUTPUT_DIR`,
`<FAMILY>_LORA_TARGET_MODULES`, `<FAMILY>_CONFIG_SUFFIXES`; the config
dataclass (a `StarterConfigMixin` subclass with the canonical field order);
`get_<family>_system_prompt`, `get_<family>_profile_overrides`,
`get_<family>_profile_description`, `summarize_<family>_config`,
`describe_<family>_starter_profiles`, `get_<family>_config`,
`create_<family>_agent_config`, `get_<family>_gspo_overrides`,
`get_<family>_gspo_config`, `create_<family>_preview`,
`load_<family>_config_file`, `write_<family>_config_file`,
`run_<run_suffix>_config`, and `finetune_<run_suffix>`. Keep the module-level
`get_config_for_task` import (tests patch it on the family module).
`tests/unit/test_starter_factory.py` enforces this contract for every family
registered in `core/model_presets.py`.

To propose a new first-class starter:

1. Create `stateset_agents/training/<family>_starter.py` with its constants,
   `validate_<family>_config`, and a `StarterSpec` (copy an existing spec,
   e.g. `kimi_k3_starter.py`); the factory generates the rest.
2. Wire it into `stateset_agents/training/__init__.py` lazy-import map.
3. Add a `@app.command("<family>-short-name")` in `stateset_agents/cli.py`.
4. Ship `examples/finetune_<family>_gspo.py` + `docs/<family>_starter.rst`.
5. (Optional) Kubernetes manifest under `deployment/kubernetes/`.
6. Add a row to the first-class table in this doc.
