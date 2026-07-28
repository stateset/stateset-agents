# A+ Final Wave Implementation Plan (Plan 4 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining A+ gaps: real finetune-surface consolidation (driver absorbs the shared flags), rate-limiter hardening, `grpo/` untangling, a genuine CPU convergence e2e test, and honest dashboard/mobile labeling.

**Architecture:** Extends `examples/finetune_gspo.py` + `examples/model_presets.py` (from Plan 3) so the per-model scripts become true forwarders; hardening lands inside existing modules; the convergence test is a slow-marked integration test using a tiny real model on a trivially learnable task.

**Tech Stack:** Python 3.10, torch/transformers tiny models, FastAPI, pytest.

## Global Constraints

- Never weaken existing passing tests; tighten only.
- Secure-by-default for any new flag; ruff (E,W,F,B,C4,UP) clean.
- Conventional commits ending `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; do not push.
- CHANGELOG entries for user-visible changes.

---

### Task 1: Unified driver absorbs the shared finetune flags; convert now-reproducible scripts to forwarders

**Files:**
- Modify: `examples/finetune_gspo.py`, `examples/model_presets.py`
- Modify: each `examples/finetune_*_gspo.py` whose behavior becomes reproducible → forwarder (deprecation print + delegate); keep any script with genuinely unique logic and record why in `examples/README.md`
- Test: extend `tests/unit/test_example_model_presets.py`

**Interfaces:**
- Driver gains the flag families the kept scripts actually share (inventory them first by reading all kept scripts): `--use-lora/--no-lora`, `--use-4bit/--use-8bit`, `--use-vllm`, `--wandb`, `--export-merged`, `--write-config PATH`, `--starter-profile {balanced,memory,quality}` (delegating to the packaged starter configs in `stateset_agents/training/*_starter.py` when the preset maps to one), `--learning-rate`, `--epochs/--steps` overrides.
- `ModelPreset` gains `starter_module: str | None` naming the packaged starter (e.g. "kimi_k3_starter") for profile delegation.
- A forwarder is ≤15 lines: deprecation print + `sys.exit(main(["--model", NAME, *sys.argv[1:]]))`.

- [ ] **Step 1: Inventory flags across kept scripts; write failing tests** — driver `--model glm5.2 --starter-profile memory --dry-run` exits 0 and the resolved config matches `get_glm5_2_config(profile="memory")`'s values; `--write-config` writes a JSON round-trippable file; forwarder scripts still exit 0 under `--dry-run`.
- [ ] **Step 2: Verify fail**
- [ ] **Step 3: Implement driver extensions; convert every script whose full CLI is now reproducible; update examples/README.md and CHANGELOG**
- [ ] **Step 4: Run `pytest tests/unit -k "example" -q`, `pytest examples/testing/ -q` — pass**
- [ ] **Step 5: Commit** `feat(examples): unified finetune driver absorbs shared flags; per-model scripts become forwarders`

---

### Task 2: Rate-limiter hardening

**Files:**
- Modify: `stateset_agents/api/middleware.py`, `stateset_agents/api/config.py` (if a knob is needed)
- Test: extend `tests/api/test_rate_limit_identity.py`

**Interfaces:**
- Credential-derived bucket keys ONLY for credentials that validate against configured API keys/JWT (reuse auth.py validation helpers); unvalidated/garbage credentials fall back to the client-IP bucket — closes the unlimited-buckets bypass.
- In-memory limiter's bucket dict hard-capped (e.g. `MAX_BUCKETS = 10_000`, evict oldest window on overflow) so unique-key floods cannot grow it unboundedly between cleanups.
- Redis limiter retries after failure: instead of permanent self-disable, record `_redis_disabled_until = now + 60s` and re-attempt after the cooldown; still log transitions once per state change, not per request. Make INCR/EXPIRE atomic via pipeline or `SET NX EX` + INCR pattern.

- [ ] **Step 1: Failing tests** — garbage `X-API-Key` values share the IP bucket; bucket dict never exceeds cap under a 20k-unique-key flood (use small cap override in test); Redis limiter re-attempts after cooldown (monkeypatched clock/redis stub).
- [ ] **Step 2: Verify fail** → **Step 3: Implement** → **Step 4: `pytest tests/api -q` green** → **Step 5: Commit** `fix(api): validate credentials before identity bucketing, cap limiter memory, redis retry with cooldown`

---

### Task 3: Untangle `grpo/rate_limiter` and scope the deprecation warning

**Files:**
- Move: `stateset_agents/api/grpo/rate_limiter.py` → `stateset_agents/api/rate_limiter.py` (git mv; `grpo/rate_limiter.py` becomes a thin re-export)
- Modify: `stateset_agents/api/middleware.py:38` import; `stateset_agents/api/grpo/__init__.py` — replace module-level `warnings.warn` with a `__getattr__`-based warning that fires only for the deprecated app surface (`service`, `router_v1`, `create_app`-style symbols), not for `rate_limiter`
- Test: `tests/api/test_grpo_deprecation_scope.py` (new) — importing `stateset_agents.api.main` (which pulls middleware) raises NO DeprecationWarning; accessing `stateset_agents.api.grpo.service` (or its equivalent public symbol) DOES

- [ ] **Step 1: Failing tests** → **Step 2: Implement move + scoped warning; fix all importers (grep)** → **Step 3: `pytest tests/api -q` green, warning count in suite output drops to 0** → **Step 4: Commit** `refactor(api): move rate_limiter out of grpo package; scope deprecation warning to the deprecated surface`

---

### Task 4: CPU convergence e2e test

**Files:**
- Test: `tests/e2e/test_gspo_convergence_tiny.py` (new)
- Modify (only if a real bug surfaces): trainer code

**Interfaces:**
- Task: single-token preference — reward 1.0 when the response contains a target token from the prompt ("Say A"), else 0.0. Tiny GPT2 (2 layers, vocab 200 or char-level tokenizer from the invariant suite's fixtures — reuse `tests/integration/test_trainer_ratio_invariants.py` plumbing).
- Train GSPO for ~30-50 steps on CPU with lr high enough to move a tiny model; assert `mean(reward over last 10 steps) > mean(reward over first 10 steps) + margin` and that the probability of the target token under the final policy strictly increased vs the initial policy (the latter is the low-variance assertion; make the reward assertion advisory/logged if flaky).
- Marked `@pytest.mark.slow` (excluded from the default fast path, runs in the nightly/benchmark workflow — add it to whichever CI job runs `-m slow`, check `.github/workflows/` for the nightly job; if none runs slow tests, add it to the benchmark-nightly workflow).
- Budget: must finish < 5 minutes on CPU; tune steps/model size to fit.

- [ ] **Step 1: Write the test (it should pass if the trainers are correct — run it 3× to check stability; if it exposes a real trainer bug, fix the trainer and note it)**
- [ ] **Step 2: Wire into nightly CI; validate YAML**
- [ ] **Step 3: Commit** `test(e2e): GSPO convergence on tiny model — policy provably improves`

---

### Task 5: Honest dashboard/mobile labeling

**Files:**
- Modify: `README.md` (project-structure/features sections mentioning dashboard/mobile, if any), `dashboard/README.md` (create if missing), `mobile/README.md` (create if missing)
- Modify: `mobile/hooks/useTrainingData.ts` — the silent mock-data fallback must surface state: expose an `isMockData`/`isDemo` flag in the hook's return and log a console warning when falling back (do NOT redesign the UI; minimal change so consumers can render a badge later)
- Modify: `dashboard/src/api.ts` — header comment stating it targets the simulator-backed `/api/lab` (auth + `API_ENABLE_TRAINING_LAB` required)

**Constraints:** No new deployment machinery — this task makes status honest, it does not ship the apps. The ship-or-archive decision stays with the user; record it as an open question in the final report.

- [ ] **Step 1: Write READMEs (status: demo, simulator-backed, not deployed; how to run locally; what would be needed to productionize)**
- [ ] **Step 2: Add the fallback flag + warning; run dashboard/mobile test suites if runnable (`npm test` where package.json defines it; skip with a note if node_modules absent)**
- [ ] **Step 3: CHANGELOG entry; commit** `docs: honest status labeling for dashboard and mobile; surface mock-data fallback`

---

## Self-Review notes

- Task 1 depends on Plan 3's merged driver; run this plan only after feat/surface-consolidation merges.
- Task 4's flakiness risk is handled by preferring the target-token-probability assertion over raw reward.
- After all tasks: re-run coverage on a healthy env if available; ratchet `fail_under` only to a measured floor.
