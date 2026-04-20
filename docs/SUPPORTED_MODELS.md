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

Each CLI command supports `--list-profiles`, `--write-config`, `--config`,
`--starter-profile {balanced,memory,quality}`, `--task`, `--json-output`, and
`--dry-run` / `--no-dry-run`.

## Reference models (examples, hosting plans, finetuning guides)

These have example scripts, Kubernetes manifests, or dedicated docs but are not
exposed as `*_starter.py` modules with profile presets.

| Model | HuggingFace ID | Surface |
|---|---|---|
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

The `stateset_agents/training/*_starter.py` modules follow a uniform shape. To
propose a new first-class starter:

1. Create `stateset_agents/training/<family>_starter.py` exporting
   `<FAMILY>_BASE_MODEL`, `<FAMILY>_SUPPORTED_VARIANTS`, profile catalog,
   `get_<family>_config`, `create_<family>_preview`, `run_<family>_config`,
   `write_<family>_config_file`, `load_<family>_config_file`.
2. Wire it into `stateset_agents/training/__init__.py` lazy-import map.
3. Add a `@app.command("<family>-short-name")` in `stateset_agents/cli.py`.
4. Ship `examples/finetune_<family>_gspo.py` + `docs/<family>_starter.rst`.
5. (Optional) Kubernetes manifest under `deployment/kubernetes/`.
6. Add a row to the first-class table in this doc.
