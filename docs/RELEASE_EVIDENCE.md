# Release evidence

This page records what was actually exercised for the current release line.
Registration, authentication, hardware allocation, training, inference, and
publication are separate claims.

## v0.42.1 and post-release hardening

Source revision: `f1edd3176b7f76d85e527a13cf18be049c16baf7`

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
