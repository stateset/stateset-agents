# GLM-5.3-Flash

StateSet Agents supports the official `zai-org/GLM-5.3-Flash` checkpoint for
text-only SFT and RL post-training through the unified GSPO preset.

```bash
pip install "stateset-agents[glm53]"
python examples/finetune_gspo.py --model glm5.3-flash --dry-run
python examples/finetune_gspo.py --model glm5.3-flash --no-dry-run
```

The model is a native multimodal `glm5_next` conditional-generation
checkpoint, not a conventional `AutoModelForCausalLM` repository. StateSet's
shared agent, SFT, GSPO, DAPO, GEPO, and VAPO loading path therefore retries
with Transformers' composite multimodal auto-model classes while keeping the
RL interaction and reward pipeline text-only. The dedicated `glm53` extra
requires Transformers 5.16 or newer, matching the checkpoint metadata.

## Architecture-aware adaptation

The preset targets both kinds of language layer in the hybrid model:

- linear attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `f_a_proj`,
  `f_b_proj`, `g_a_proj`, `g_b_proj`, and `b_proj`;
- sparse attention: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, and
  `kv_b_proj`.

Those names were checked against the official weight index on 2026-08-26.
The vision tower and the 288-expert MoE tensors are deliberately excluded;
adapting every expert by leaf name would create an impractically large LoRA.
The published checkpoint is already FP8, so the preset does not stack 4-bit or
8-bit loading on top of it.

## Native multimodal inference

StateSet's current RL trajectory schema is text-first. For image or video
inputs, use the official `AutoProcessor` + `AutoModelForMultimodalLM` flow, or
serve the model through vLLM/SGLang's OpenAI-compatible endpoint and connect
that endpoint to your application. Do not pass image dictionaries to the
text-only RL tokenizer and assume the visual inputs were consumed.

The official model card documents 320B total parameters, 18B active
parameters, a 1,048,576-token maximum position setting, and native text,
image, and video input. Plan multi-GPU storage and serving capacity for the
roughly 328 GB repository even though active inference compute is much lower.

Sources: [official model card](https://huggingface.co/zai-org/GLM-5.3-Flash),
[official configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json).
