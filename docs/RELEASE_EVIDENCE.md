# Release evidence

This page records what was actually exercised for the current release line.
Registration, authentication, hardware allocation, training, inference, and
publication are separate claims.

## v0.42.5 release evidence

Five-algorithm execution revision:
`062a420b6cbdb4b532115a6c6500041ac1c68aa2`.

| Surface | Result | Evidence |
|---|---|---|
| Default suite | Passed | 4,729 passed, 6 skipped, 0 failed |
| Static typing | Passed | 304 source files, 0 issues |
| Lint, format, hygiene | Passed | Ruff, Black (720 files), isort, and repository hygiene |
| Distribution | Passed | Wheel and sdist built; Twine metadata passed; isolated wheel import reported 0.42.5 and CLI help passed |
| Five-algorithm matrix | Passed | [15 measured CUDA runs](../benchmark_results/algorithm_comparison/report/comparison.md): GRPO, GSPO, DAPO, VAPO, and GEPO across seeds 42, 1337, and 2026 |
| Benchmark provenance | Passed | Pinned model/data, exact shared/objective configs, RTX 5080/CUDA 12.9, external timers, actual completion counts, and 15 normalized-policy hashes |
| Rejected diagnostic | Retained | [First full attempt](../benchmark_results/algorithm_comparison/diagnostics/d883f8a-full-attempt1/README.md) preserved three deterministic VAPO OOM failures |
| VAPO memory correction | Passed | Three full-shape runs completed at 6,078 MiB mean peak VRAM after the rejected attempt reached 15.28/15.47 GiB |
| RunPod cleanup | Passed | Every validation/matrix pod terminated after retrieval; final account inventory contained zero pods |

The five-algorithm report is descriptive and scoped to its four-step,
0.5B-model protocol. It closes the matched native-algorithm execution gate; it
does not establish 8B performance, multi-node scaling, comparison with verl,
NeMo RL, or OpenRLHF, independent reproduction, or broader ecosystem maturity.

## v0.42.3 release evidence

Release readiness revision: `dbf4efe3b90d507aac0a4a9051f8ba939922a826`

Post-tag scaling/recovery evidence revisions:
`722f7e9fdafceec48723dc4392a212418cba9f2b` and
`52df33fa00d7eac4b2973fdeb4a5446f6278a8b1`. The corrected strong-scaling
harness revision is `f0b90809b1dd5d75d2249a1237a371e87ff6a81b`; evidence retention and
execution-contract validation landed in `75367e2`, and the full competitive
roster gate landed in `5743704`.

| Surface | Result | Evidence |
|---|---|---|
| Default and coverage suite | Passed | 4,711 passed, 6 skipped; 62.71% coverage against 61% floor |
| Static typing | Passed | 304 source files, 0 issues |
| Lint, format, repository hygiene | Passed | Ruff, Black, isort, and `scripts/check_repo_hygiene.py` |
| Security | Passed | Bandit policy passed; Safety found 0 vulnerabilities in `requirements-dev-lock.txt` |
| Distribution | Passed | Wheel and sdist built; Twine metadata passed; package smoke reported 0.42.3 |
| StateSet vs direct TRL | Passed | [Three seeds on one NVIDIA A40](../benchmark_results/framework_comparison/report/comparison.md); equivalent throughput, wall time, and VRAM for the exact four-step protocol, with all quality outcomes retained |
| Benchmark provenance | Passed | Harness commit `4173eee7be7187c0583390953b8ad79b55fb954f`; pinned model/data; exact config digest; six artifact hashes |
| RunPod cleanup | Passed | Every benchmark pod terminated after evidence retrieval; final read-only canary observed 0 pods and 0 cleanup leases |
| 1/2/4/8-GPU weak scaling | Passed | [Three seeds per topology](../benchmark_results/scaling/report/scaling.md); 8.080× throughput at 8 GPUs and 101.0% weak-scaling efficiency |
| 1/2/4/8-GPU strong scaling | Passed | [Three seeds per topology](../benchmark_results/scaling/strong/report/scaling.md); fixed 196,608-sample effective batch, 6.642× throughput and 83.0% efficiency at 8 GPUs |
| Short-work strong-scaling diagnostic | Failed | [Negative matrix retained](../benchmark_results/scaling/diagnostics/f6e7478-strong/report/scaling.md); the 2,048-sample workload measured only 0.188s at one GPU and reached 2.9% 8-GPU efficiency |
| Fault-injection matrix | Passed | [Nine CUDA runs](../benchmark_results/reliability/report.json); exact replay, zero lost/duplicate updates, completion, cleanup |
| River live canary | Passed | 2026-08-27 read-only health/capabilities; 0 billable resources |
| RunPod live canary | Passed | 2026-08-27 read-only inventory/cleanup; 0 billable resources |
| Fireworks live canary | Skipped | `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID` are unavailable; [skipped result retained](../benchmark_results/provider_canaries/fireworks-2026-08-27.json) |

The four-step shootout is an orchestration-parity test, not evidence that four
steps improve GSM8K quality. Its null/negative quality deltas are retained.
Weak- and strong-scaling claims are reported separately. The corrected strong
protocol holds total work fixed and does not erase the failed short-work
diagnostic. Neither single-host matrix establishes multi-node performance.

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
