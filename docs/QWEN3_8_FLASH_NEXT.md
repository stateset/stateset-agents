# Qwen3.8-Flash-Next

StateSet Agents supports the official `Qwen/Qwen3.8-Flash-Next` checkpoint for
text-only SFT and RL post-training through the unified GSPO preset.

```bash
pip install "stateset-agents[qwen38next]"
python examples/finetune_gspo.py --model qwen3.8-flash-next --dry-run
python examples/finetune_gspo.py --model qwen3.8-flash-next --no-dry-run
```

The model is a native multimodal `qwen4_exp` conditional-generation
checkpoint. StateSet's shared composite loader handles the causal-auto-model
fallback for agents, SFT, GSPO, DAPO, GEPO, and VAPO. RL trajectories and
rewards remain text-only; image and video inputs require `AutoProcessor` or an
OpenAI-compatible multimodal serving endpoint.

## Architecture-aware adaptation

The preset targets the language model's two attention systems:

- Gated DeltaNet: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, and
  `out_proj`;
- Qwen Sparse Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`, and
  `index_qk_proj`.

These names were checked against the official weight index on 2026-08-26.
The vision encoder and the 512-expert MoE tensors are excluded. The sparse
attention leaf names also occur in the single MTP layer; PEFT's leaf-name
matching therefore adapts that layer as well. This is recorded explicitly
rather than presented as language-layer-only targeting.

The default preset uses 4-bit loading of the original checkpoint for adapter
training. The separate `Qwen/Qwen3.8-Flash-Next-FP8` repository is intended
for efficient inference and should not be treated as the adapter-training
base. Even with only 6B parameters active per token, storage must account for
the 125B main model, 51B n-gram embeddings, and 4B MTP component.

## Serving

The official deployment examples use 262,144 tokens and four-way tensor
parallelism with these parsers:

```bash
vllm serve Qwen/Qwen3.8-Flash-Next \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

The checkpoint thinks by default. For deterministic evaluation, record
whether `enable_thinking` and `preserve_thinking` are enabled rather than
mixing both modes in one benchmark.

Sources: [official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
[official repository](https://github.com/QwenLM/Qwen3.8-Flash-Next).
