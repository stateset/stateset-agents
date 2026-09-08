# Live vLLM rollout weight-sync check

Retained records from `benchmarks/vllm_rollout_check.py` on one RunPod NVIDIA A40
(2026-09-08). Each run attaches a `VLLMGenerator` to a `MultiTurnAgent` holding
the same Hugging Face model, samples four turns through the engine, takes three
real GRPO token-path optimizer steps with `old_logprobs_source="sampler"`, and
measures the engine-vs-policy log-prob gap of the sampled tokens (nats, max /
mean over 31 tokens)
before the step, after the step with the engine still stale, and after
`VLLMGenerator.sync_weights`.

| Record | Passed | Before step | After step, before sync | After sync | Syncs | Sync error |
|---|---|---|---|---|---|---|
| `full.json` (full weights, Adam lr 5e-4) | True | 0.172 / 0.037 | 43.01 / 19.49 | 0.291 / 0.084 | 1 | - |
| `peft.json` (LoRA r=8 on q_proj/v_proj) | True | 0.172 / 0.037 | 3.88 / 0.62 | 0.105 / 0.024 | 1 | - |
| `peft-attempt1-failed.json` (before the `.base_layer` fix) | False | 0.172 / 0.037 | 3.88 / 0.62 | 3.879 / 0.618 | 0 | AttributeError: QKVParallelLinear has no attribute `base_layer` |

Pass criteria: the sync succeeds, the after-sync gap is within max(3x the
before-step gap, 0.05) nats, the stale gap exceeds the after-sync gap, and a
turn generated after the sync comes from the engine at policy version 1.

Environment: vLLM 0.28.0, torch 2.13.0+cu130,
transformers 5.16.1, stateset-agents
0.51.0 at harness `ea4dc2c509ec`
(`peft.json`) and `e5a7bbbf37f0` (`full.json`, `peft-attempt1-failed.json`),
model `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf`, bf16, in-process V1
engine core (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), engine model reached as
`Qwen2ForCausalLM`.

The first PEFT attempt failed inside the engine's `load_weights`
(`QKVParallelLinear has no attribute base_layer`): a merged `PeftModel` names
its weights `<module>.base_layer.weight`. The agent reported the backend
stale with that error instead of sampling from the old policy, which is the
designed failure mode; the name mapping was fixed and the check re-run.

Provider records: `runpod-provider-attempt1.json` (pod `e7aq6pqpaz8eh8`,
$0.11) and `runpod-provider.json` (pod `hd5h0hbqo1lup2`,
$0.06); a first pod that failed on a script bug before
loading any model cost $0.08 (ledger only). Total: about $0.25.
