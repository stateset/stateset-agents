# GPU evidence plan (costed, for approval)

Purpose: close the "scale" gap the independent breadth reviews keep citing
(scale/infra C+): verified sharded and multi-node RL, a scalable rollout
engine, and GPU evidence for the beta components. Nothing here is launched
without explicit approval of a phase and its ceiling.

Prices: RunPod catalog, queried 2026-10-27 (USD per GPU-hour):

| GPU | Secure | Community |
|---|---|---|
| H100 80GB HBM3 (SXM) | 3.99 | 2.69 |
| A100 80GB (SXM/PCIe) | 1.79 | 1.19–1.39 |
| L40S 48GB | 1.09 | 0.79 |

Ceilings below use secure-cloud prices (community pods have been less
reliable for networking) and include a ~30% retry margin. Every run uses the
existing launcher safeguards (worst-case price check before launch,
unconditional termination, cost ledger) and retains its evidence under
`benchmark_results/`.

## Phase 0: graduate beta components on one GPU (existing code)

| Run | Hardware | Time | Ceiling |
|---|---|---|---|
| QLoRA SFT (4-bit, CUDA) on a 1.5B model, loss + adapter round trip | 1x L40S | 0.5 h | $1 |
| Preference DPO/KTO and LoRA reward model on a 0.5-1.5B model, 3 seeds | 1x A100 80GB | 2 h | $5 |
| Distillation (logit KD + on-policy GKD), 1.5B teacher -> 0.5B student, 3 seeds | 1x A100 80GB | 2 h | $5 |
| Tool-use RL (calculator + math verifier) on Qwen-0.8B/1.5B, 3 seeds | 1x A100 80GB | 2 h | $5 |
| lm-eval capability-regression hook against a real checkpoint | 1x L40S | 1 h | $2 |

Phase 0 ceiling: **~$25**. No new code required.

## Phase 1: DDP under NCCL on 8 GPUs (existing code)

GRPO and GSPO token-path training on a 1.5B model across 8 GPUs, verifying the
cross-rank update contract (today verified only on CPU/Gloo), with throughput
and loss parity against a single-GPU baseline.

| Run | Hardware | Time | Ceiling |
|---|---|---|---|
| 8-GPU DDP GRPO + GSPO, parity + throughput | 8x H100 SXM | 1.5 h | **~$65** |

## Phase 2: sharded RL at 7-8B (needs FSDP2 code first)

Requires implementing FSDP2 (fully_shard) for the token-path trainers,
including sharded reference/old-policy models; verified on CPU/Gloo first.

| Run | Hardware | Time | Ceiling |
|---|---|---|---|
| FSDP2 GRPO on a 7-8B model, 3 seeds of a short learning run | 8x H100 SXM | 3 h | **~$125** |

## Phase 3: multi-node (needs Phase 2)

| Run | Hardware | Time | Ceiling |
|---|---|---|---|
| 2-node FSDP2 GRPO, 7-8B, retained run + throughput | 16x H100 SXM | 2 h | **~$170** |

## Phase 4: scalable rollout engine (needs vLLM server-mode code first)

Requires vLLM server mode with NCCL weight broadcast (TP >= 2) and wiring the
existing async rollout runtime into the multi-turn trainer.

| Run | Hardware | Time | Ceiling |
|---|---|---|---|
| vLLM TP=2 rollouts + weight sync + async off-policy GRPO, 1.5-7B | 4x H100 SXM | 2 h | **~$85** |

## Totals

| | Ceiling |
|---|---|
| Phase 0 + 1 (existing code, can run now) | **~$90** |
| Phases 2-4 (after the code lands) | **~$380** |
| All phases | **~$470** |

Standard agent benchmarks (tau-bench, BFCL) additionally need API spend for
their user simulators/judges, estimated at $20-50 per paired base-vs-trained
comparison, approved separately.

Recommended order: approve Phase 0 + 1 now (~$90, existing code); start the
FSDP2 and vLLM server-mode code tracks in parallel; approve Phases 2-4 only
after their code passes CPU verification.
