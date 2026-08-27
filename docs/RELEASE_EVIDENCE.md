# Release evidence

This page records what was actually exercised for the current release line.
Registration, authentication, hardware allocation, training, inference, and
publication are separate claims.

## v0.42.3 release evidence

Release readiness revision: `dbf4efe3b90d507aac0a4a9051f8ba939922a826`

Post-tag scaling/recovery evidence revisions:
`722f7e9fdafceec48723dc4392a212418cba9f2b` and
`52df33fa00d7eac4b2973fdeb4a5446f6278a8b1`.

| Surface | Result | Evidence |
|---|---|---|
| Default and coverage suite | Passed | 4,677 passed, 6 skipped; 62.71% coverage against 61% floor |
| Static typing | Passed | 304 source files, 0 issues |
| Lint, format, repository hygiene | Passed | Ruff, Black, isort, and `scripts/check_repo_hygiene.py` |
| Security | Passed | Bandit policy passed; Safety found 0 vulnerabilities in `requirements-dev-lock.txt` |
| Distribution | Passed | Wheel and sdist built; Twine metadata passed; package smoke reported 0.42.3 |
| StateSet vs direct TRL | Passed | [Three seeds on one NVIDIA A40](../benchmark_results/framework_comparison/report/comparison.md); equivalent throughput, wall time, and VRAM for the exact four-step protocol, with all quality outcomes retained |
| Benchmark provenance | Passed | Harness commit `4173eee7be7187c0583390953b8ad79b55fb954f`; pinned model/data; exact config digest; six artifact hashes |
| RunPod cleanup | Passed | Every benchmark pod terminated after evidence retrieval; final read-only canary observed 0 pods and 0 cleanup leases |
| 1/2/4/8-GPU weak scaling | Passed | [Three seeds per topology](../benchmark_results/scaling/report/scaling.md); 8.080× throughput at 8 GPUs and 101.0% weak-scaling efficiency |
| Fixed-global-batch strong scaling | Failed | [Negative matrix retained](../benchmark_results/scaling/diagnostics/f6e7478-strong/report/scaling.md); 8-GPU efficiency was 2.9%, so no strong-scaling claim is made |
| Fault-injection matrix | Passed | [Nine CUDA runs](../benchmark_results/reliability/report.json); exact replay, zero lost/duplicate updates, completion, cleanup |
| River live canary | Passed | 2026-08-27 read-only health/capabilities; 0 billable resources |
| RunPod live canary | Passed | 2026-08-27 read-only inventory/cleanup; 0 billable resources |
| Fireworks live canary | Skipped | `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID` are unavailable; [skipped result retained](../benchmark_results/provider_canaries/fireworks-2026-08-27.json) |

The four-step shootout is an orchestration-parity test, not evidence that four
steps improve GSM8K quality. Its null/negative quality deltas are retained.
The scaling result is explicitly weak scaling; it does not supersede the
failed strong-scaling diagnostic or establish multi-node performance.

## v0.42.2 release evidence

Live verification revision: `f1edd3176b7f76d85e527a13cf18be049c16baf7`

| Surface | Result | Evidence |
|---|---|---|
| Default test suite | Passed | 4,615 passed, 6 skipped, 0 failed |
| Static typing | Passed | 303 source files, 0 issues |
| Repository hygiene | Passed | `scripts/check_repo_hygiene.py` |
| River authentication/capabilities | Passed | Read-only canary at 2026-08-26T20:53:47Z; 0 billable resources |
| RunPod authentication/cleanup | Passed | Read-only canary at 2026-08-26T21:39:39Z; 0 pods and 0 cleanup leases |
| RunPod SFT | Passed | [GPU Verify run 33015161235](https://github.com/stateset/stateset-agents/actions/runs/33015161235): Qwen3.5-0.8B QLoRA, 2/2 held-out checks, 21,694,976-byte adapter |
| RunPod RL | Passed | Same run: real CUDA GSPO, target probability 0.0000281 → 0.124616 in 40 steps |
| RunPod cleanup | Passed | Both jobs terminated their pods; post-run canary observed 0 pods |
| Fireworks live canary | Not run | `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID` are not available |
| PyPI v0.42.1 | Blocked | [Publish run 33011010658](https://github.com/stateset/stateset-agents/actions/runs/33011010658): PyPI rejected the OIDC identity as `invalid-publisher` |

The default suite excludes tests marked `slow`, `gpu`, and `performance`.
GPU evidence is therefore listed independently and linked to its retained
workflow logs and adapter artifact.
