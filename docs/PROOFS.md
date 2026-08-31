# The proof dashboard

Every headline claim this project makes, mapped to the evidence behind it —
what kind of proof it is, where it lives, and how (or whether) it is kept
true over time. The categories are strict:

- **Re-proved automatically** — a scheduled CI job runs the claim again and
  goes red if it stops being true.
- **Live-verified** — it happened on rented hardware at least once, with the
  artifact or log retained. True as of the date shown; not automatically
  re-run.
- **Unit-pinned** — the behaviour is enforced by tests on every commit, but
  has never needed (or cannot have) a GPU.
- **Unverified** — labelled as such wherever it is mentioned. We say so
  rather than imply otherwise.

| Claim | Category | Evidence |
|---|---|---|
| The loop raises a ceiling: base 0/12 → gen‑1 2/12 → gen‑2 10/12 | Live-verified (2026‑08‑17, reproduced twice) | [`FLYWHEEL_HEADROOM.md`](FLYWHEEL_HEADROOM.md) — pods, costs, and both trainings' identical 10/12 |
| Fine-tuning works end to end on rented GPUs | Live-verified, many runs | [`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md); adapters + `stateset_manifest.json` under `outputs/` |
| `serve-remote` answers authenticated HTTPS requests | Live-verified (2026‑08‑17, twice: hand-driven + shipped CLI with adapter) | CHANGELOG v0.31.0; flashinfer patch + arm-precedence fix pinned by tests |
| The RL core learns on real hardware (GSPO 2.8e‑05 → 0.125) | Re-proved weekly* | [`gpu-verify.yml`](../.github/workflows/gpu-verify.yml) `rl-live-smoke` |
| Multi-GPU sharding (`--gpu-count`) | Live-verified (63GB across 2×48GB) | CHANGELOG v0.28.0 — device-map log `0=24 module(s), 1=36 module(s)` |
| Single-host DDP strong scaling | Live-verified (2026-08-27, three seeds) | [`BENCHMARKS.md`](BENCHMARKS.md#single-node-ddp-weak-and-strong-scaling); fixed 196,608-sample work, 6.642× at 8× RTX 5080, 83.0% efficiency; 12 raw evidence rows retained |
| Curation precision/recall 1.000/1.000 | Re-proved on every CI run | `make benchmark-loop`, floors ratcheted at 0.95 |
| A failed eval gate preserves its adapter | Unit-pinned (from a live incident) | `test_failed_job_still_attempts_fetch` — docstring cites the lost 10/12 adapter |
| Unknown cost is refused, never treated as $0 | Unit-pinned | `tests/unit/test_remote_ledger.py` budget tests |
| Pods terminate on every exit path, incl. client death | Live-verified + unit-pinned | self-destruct fired with the client force-killed (v0.26.0); armed-first ordering tested |
| Multi-adapter serving + SSE streaming over the proxy | Live-verified (2026‑08‑18), mechanics only | `/v1/models` listed both adapters; SSE chunks flowed; **see LoRA caveat below** |
| **vLLM applies hybrid-Qwen3.5 LoRA adapters** | **DISPROVEN (2026‑08‑18)** | Greedy adapter output byte-identical to base on training-format prompts; vLLM logs 'Loaded' with no error. Upstream limitation: the hybrid `linear_attn` target names never match. `chat-remote` (transformers+peft) remains the verified way to talk to these fine-tunes; earlier 'adapter answers differed' observations were temperature noise |
| River AI provider | **Live-verified** (2026‑08‑18); scheduled canary ready | `train-remote --provider river` trained Qwen3.5‑9B for real (session → LoRA model → train step → `river://` checkpoint, lineage manifest written) and the training EFFECT is proven: 140 TechNest rows × 3 epochs (210 steps), then **3/3 held-out tickets** answered from the checkpoint with the canonical resolutions, persona signature, and echoed ticket numbers. [`provider-canary.yml`](../.github/workflows/provider-canary.yml) rechecks health/capabilities when `RIVER_API_KEY` is configured. |
| CoreWeave CKS training and Dedicated Inference | **Unit-pinned; live certification pending** | Kubernetes Job construction, GPU selectors/counts, active deadlines, Secret references, durable state, S3 artifact round-trip/cleanup, read-only RBAC canary, BYOW gateway/deployment payloads, rollback, status, and deletion are covered in `tests/unit/test_remote_coreweave_executor.py` and `tests/unit/test_remote_cloud_artifacts.py`; no live CKS or inference claim is made. |
| Nebius Serverless AI jobs and endpoints | **Unit-pinned; live certification pending** | Official CLI construction, SecretStash selectors, durable reconnects, state mapping, S3 artifact round-trip/cleanup, cancellation, vLLM endpoint lifecycle, and unsupported-cost/scaling failures are covered in `tests/unit/test_remote_nebius_executor.py` and `tests/unit/test_remote_cloud_artifacts.py`; no live job or endpoint claim is made. |
| `serve-remote --merge` serves hybrid fine-tunes for real | **Live-verified** (2026‑08‑18, sixth attempt) | Merged Qwen3.5-0.8B + TechNest adapter answered greedy training-format tickets in full persona over the proxy; on-pod pre/post-merge probe enforced the effect. The path there: composite architecture must be loaded as itself, text-trained adapter keys remapped (probe delta 0.0 before, real deltas after), processor artifacts saved, merge in an isolated venv |
| RL reward hacking is detectable and fixable in-platform | Live-observed + fixed (2026‑08‑18) | Energy domain: v1 reward Goodharted (reward ↑ 0.67→0.84, eval ↓ 6→4/12, exploit = dropped resolutions); completeness bonus restored monotone-ish improvement to 7/12 on the A/B rerun. Objective and eval are separate by design — that separation is what caught it |
| **Distillation breaks walls self-training cannot** | Live-verified (2026‑08‑19) | The 9B that walled at 9/12 self-harvesting (SFT regressed, RL flat) reached **11/12 in one generation** trained on the 35B teacher's harvest (93/96 kept, 97%); gen-2 (9/12) honestly discarded by the plateau stop. Rent wisdom, deploy cheap — through River, zero machines |
| Tool calls are verified deterministically | **Live-verified** (2026‑08‑19) | A ladder-trained 9B scored 5/12 on tool-gated episodes — prose alone never passes; the action parses, names the tool, and carries the right args, or the turn fails |
| **Unchecked turns pollute the harvest** | **Live-observed and fixed** (2026‑08‑19) | Turns with no tool requirement emitted invented tools (`suppress_dispatch`) and malformed multi-object blocks; 113 such episodes trained a model that stopped emitting valid actions entirely (0/12 greedy, from 5/12). Episodes now declare `known_tools` and any junk block anywhere fails the episode. **Guardrail verified by rerun**: same 5/12 start, harvest tightened 59% → 51% (clean episodes only, 7/24 scripts now yielding nothing), trained model held **5/12 instead of collapsing to 0/12** — the collapse is fixed; no lift at this difficulty, honestly plateau-stopped |
| The rarity controller targets the operating window | **Live-verified** (2026‑08‑19) | `--target-harvest-rate 0.6` probed three temperatures, chose 0.7, and the real harvest landed at **59%** — the measured regime hit on the first live attempt |
| The multi-turn episode flywheel runs end to end | Live-verified (2026‑08‑18) | Two-turn scripts scored per turn (carryover objective: turn 2 demands the reference the user never repeats); 144 episode rollouts, 93 harvested, gen-2 trained on conversations, plateau-stop kept gen-1 (9/12 vs 8/12). No lift at rung 2 (9/12 baseline, 65% harvest — nothing rare to amplify); at rung 3 (three turns, final-turn summary of all actions, 40% refusals) the doctrine paid: **8/12 → 12/12 in one generation** on 115 harvested conversations — the flywheel's first multi-turn ceiling-raise, and a perfect-score stop. **The ladder works as a curriculum**: the rung-3 adapter transferred UPWARD to rung 4 (four turns, 50% refusals) at 11/12 — above the untrained model's rung-3 score — and one more wheel-turn topped it off at 12/12. **Rung 5 found the wall, honestly**: baseline 9/12 with an 83% harvest rate, gen-2 scored 7/12, plateau-stop kept rung 4. The full curve pins the operating regime: the wheel lifts near ~60% harvest with headroom, stalls when temperature success is too common to be informative |
| The RL flywheel (`--algorithm cispo`) trains for real | **Live-verified** (2026‑08‑18) | First run: 7/12 → 10/12 by round 1, mean reward monotone 0.72 → 0.89 over 4 rounds, zero infrastructure. Honest head-to-head on the same depth-3 refusal kit: SFT reached 11/12 — RL is verified at parity-ish with untuned knobs, NOT yet proven superior; both methods retain one refusal violation |
| The difficulty ladder restores headroom | Live-verified (2026‑08‑18) | The depth-2-perfect (12/12) model scores 7/12 at depth-3 + refusals; one SFT wheel-turn climbs rung 2 to 11/12, the residual failure being exactly a refusal violation |
| `stateset-agents flywheel` (the loop as one command) | **Live-verified** (2026‑08‑17) | [`FLYWHEEL_DOMAIN2.md`](FLYWHEEL_DOMAIN2.md) — full multi-generation run: 0/12 → 6/12 → 7/12 for $2.60 |
| The ceiling-raise replicates across domain and scale | Live-verified (2026‑08‑17/18, **four model/substrate/domain combinations**) | RunPod 0.8B IT-helpdesk: 0/12 → 7/12 and 0/12 → 11/12; River 9B IT-helpdesk: 7/12 → 11/12 (plateau-stop); River **35B MoE** travel-concierge: 7/12 → **12/12 in one generation** (perfect-score stop, harvest rate 74%) |

\* **Honesty note on "weekly":** `RUNPOD_API_KEY` and `RIVER_API_KEY` are
configured in GitHub Actions. Scheduled GPU verification passed on 2026-08-17
and 2026-08-24. The 2026-08-31 run failed before allocation when RunPod's pod
creation endpoint returned HTTP 500 after bounded retries; that red run remains
visible and created no training claim. "Weekly" therefore means an automated,
fail-closed schedule with retained successes and failures, not an assertion
that the most recent provider attempt was green.

PyPI rejected the repository's OIDC identity as `invalid-publisher` for
v0.43.0, so that release was uploaded with a scoped project token and its live
digests were compared byte-for-byte with the validated local artifacts. The
same scoped token is now configured as the workflow fallback. OIDC remains a
desirable owner-side hardening step. Fireworks certification still requires
`FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID`; their absence is reported as a
strict canary failure rather than silently skipped.
