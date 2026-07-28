# Surface Consolidation Implementation Plan (A+ push, Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the copy-paste example sprawl into a parameterized model registry, de-duplicate the docs corpus, fix misplaced/duplicated test files, and wire the getting-started smoke into CI.

**Architecture:** Introduce one `examples/finetune_gspo.py --model <preset>` driver backed by a preset registry (`examples/model_presets.py`) that captures what today differs between the ~15 `finetune_*_gspo.py` clones. Old scripts become 5-line forwarders for one release (deprecation notice), then die. Docs: archive dev-journal artifacts, merge the four comparison docs into one. Nothing here touches library code except test-file moves.

**Tech Stack:** Python 3.10, pytest, Sphinx docs, GitHub Actions.

## Global Constraints

- Never delete a doc/example without either archiving to `docs/archive/` / `examples/archive/` or replacing it with a forwarder; CHANGELOG entry for every removal.
- `examples/README.md` must list every non-archived example after this plan (a test enforces it).
- Ruff clean; conventional commits ending `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; do not push.
- Do not break `tests/unit/test_docs_onboarding.py` and friends; extend them.

---

### Task 1: Model-preset registry + unified finetune driver

**Files:**
- Create: `examples/model_presets.py`, `examples/finetune_gspo.py`
- Test: `tests/unit/test_example_model_presets.py` (new)

**Interfaces:**
- `examples/model_presets.py`: `@dataclass(frozen=True) ModelPreset` with fields covering everything that varies across the existing `finetune_*_gspo.py` scripts and `*_config.py` files — enumerate by diffing them first (expected: `model_id`, `tokenizer_id`, `lora_target_modules`, `max_prompt_length`, `max_completion_length`, `learning_rate`, `num_generations`, `bf16/quantization flags`, `chat_template_override`, `notes`). `PRESETS: dict[str, ModelPreset]` with one entry per currently-supported model (kimi-k3, kimi-k2.5, kimi-k2.6, glm5.1, glm5.2, qwen3, qwen3.5-0.8b, qwen3.5-27b, gemma3, gemma4-31b, llama3, mistral). Values copied faithfully from the existing per-model scripts — where two scripts for the same model disagree (`finetune_kimi_k2_5_gspo.py` vs `finetune_kimi_k25_gspo.py`), prefer the newer file (git log) and record the discrepancy in the preset's `notes`.
- `examples/finetune_gspo.py`: argparse CLI `--model <preset-name> [--list-models] [--dry-run]` reproducing the common training flow of the existing scripts (agent+reward+GSPOTrainer setup); `--dry-run` builds everything with `use_stub_model=True` and exits 0 without training.

- [ ] **Step 1: Diff the clone family to extract the varying fields; write failing tests** — every preset name in `PRESETS` round-trips through `finetune_gspo.py --model X --dry-run` (subprocess, exit 0, stub backend); `--list-models` prints all preset names.
- [ ] **Step 2: Verify fail**
- [ ] **Step 3: Implement registry + driver**
- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Commit** `feat(examples): parameterized GSPO finetune driver with model preset registry`

---

### Task 2: Convert clone scripts to forwarders; archive stale examples; complete examples/README

**Files:**
- Modify: each `examples/finetune_*_gspo.py` clone → forwarder (prints a deprecation line, then `main(["--model", "<preset>"])` from the new driver); delete each `examples/*_config.py` clone whose content is absorbed into presets
- Move to `examples/archive/`: redundant demo variants — `enhanced_framework_demo.py`, `enhanced_framework_showcase.py`, `ultimate_customer_service_demo.py`, `enhanced_customer_service.py`, `enhanced_grpo_demo.py` (keep one canonical customer-service example and one GRPO showcase; pick the ones examples/README already documents)
- Move: `examples/test_kimi_k25.py` → delete (duplicate of `tests/integration/test_kimi_k25.py`); `tests/test_kimi_k25_integration.py` → delete (duplicate of `tests/integration/test_kimi_k25_integration.py`) — verify duplicates are true duplicates first (diff); if they drifted, merge into the tests/integration copy
- Modify: `examples/README.md` — every remaining top-level example listed with one line
- Test: `tests/unit/test_examples_readme_complete.py` (new) — walks `examples/*.py` (non-archive) and asserts each filename appears in `examples/README.md`

- [ ] **Step 1: Write the README-completeness test (failing)**
- [ ] **Step 2: Do the conversions/moves/deletes; update README + CHANGELOG**
- [ ] **Step 3: Run the new test + `pytest tests/unit -k "example or docs" -q` + CI example job locally (`pytest examples/testing/ -q`) — pass**
- [ ] **Step 4: Commit** `refactor(examples): forward clone finetune scripts to unified driver; archive redundant demos; complete README index`

---

### Task 3: Docs consolidation

**Files:**
- Create: `docs/COMPARISONS.md` (single merged comparison: vs TRL, vs LLM frameworks, vs traditional RL — merge content from `docs/COMPARISON_TRL.md`, `docs/COMPARISON_LLM_FRAMEWORKS.md`, `docs/COMPARISON_TRADITIONAL_RL.md`, existing `docs/COMPARISONS.md`)
- Move to `docs/archive/`: the three superseded comparison files plus dev-journal artifacts `docs/ENHANCEMENTS_SUMMARY.md`, `docs/FRAMEWORK_ENHANCEMENT_SUMMARY.md`, and root `GYM_INTEGRATION_COMPLETE.md`
- Modify: any Sphinx toctree / README links referencing moved files (grep before moving)
- Test: extend `tests/unit/test_docs_onboarding.py` (or a new `test_docs_structure.py`) asserting the archived filenames no longer exist at their old paths and `docs/COMPARISONS.md` contains all three comparison-section headers

- [ ] **Step 1: Failing structure test**
- [ ] **Step 2: Merge + move + fix links; CHANGELOG entry**
- [ ] **Step 3: Build docs (`make -C docs html` or the CI equivalent) — no broken-ref warnings introduced; tests pass**
- [ ] **Step 4: Commit** `docs: merge comparison docs, archive dev-journal artifacts`

---

### Task 4: CI wiring + deferred-minor cleanup from Plan 1

**Files:**
- Modify: `.github/workflows/ci.yml` — add a job (or step in the examples job) running `examples/getting_started/smoke.sh` against the **source tree** (not PyPI: export PYTHONPATH or `pip install -e .` first; read the script to see what it needs); keep the existing PyPI-based `make getting-started-smoke` as-is for release checks
- Modify: `tests/unit/test_advanced_trainers.py::test_compute_gepo_coefficient` — rewrite to pass log-probs and assert against the linear-space formula (deferred minor from Plan 1 Task 2)
- Modify: `stateset_agents/training/gspo_token_trainer.py` — normalize token loss by response length instead of full padded width (deferred minor from Plan 1 Task 4), updating its behavioral test expectations if needed (tighten, don't weaken)
- Modify: `stateset_agents/training/loss_computation.py` — fix the `_estimate_policy_entropy = compute_entropy_bonus` alias's "backwards-compatible" comment (signatures differ; only caller checks `callable`)
- Modify: `CHANGELOG.md` — note that `rescore_old_log_probs=True` (new GSPO default) requires an HF agent model/tokenizer even in vLLM deployments
- Test: existing suites

- [ ] **Step 1: Rewrite the GEPO coefficient test (fails against nothing — it's a tightening; verify it passes), fix gspo_token normalization TDD-style**
- [ ] **Step 2: Wire CI step; validate workflow YAML (`actionlint` if available, else careful review)**
- [ ] **Step 3: Full fast suite green**
- [ ] **Step 4: Commit** `ci: run getting-started smoke from source; test: tighten GEPO coefficient test; fix(gspo-token): normalize by response length`

---

## Self-Review notes

- Coverage ratchet: after Plans 1–3 land, re-measure coverage and raise `fail_under` to the new floor per the documented ratchet policy (do this in Plan 3's final task if the number moved).
- Dashboard/mobile ship-or-archive remains a product decision for the user — flagged in the final report, not unilaterally executed.
