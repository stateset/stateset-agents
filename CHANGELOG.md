# Changelog

All notable changes to the StateSet RL Agent Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.13.2] - 2026-05-18 — Second PhD-reviewer round (calibration + scoping fixes)

Addresses every item from the second reviewer pass. Patch bump: no API removed, additions are doc clarifications, one framework warning, and a notebook extension. Items map 1:1 to the reviewer's numbered list.

### Added

- **§1.0 Design Philosophy preface** — the five principles previously only in §14 now lead §1, anchoring the rest of the document. §14 retains the detailed treatment.
- **§11.7 in-family-judge-bias disclosure** — explicit acknowledgment that Qwen-trainee + Qwen-judge has known in-family bias. Points readers at the new `CROSS_FAMILY_JUDGE_MODEL` knob in the reference notebook.
- **`CROSS_FAMILY_JUDGE_MODEL` constant** in `notebooks/customer_support_3seed_judge.ipynb` — opt-in second judge (Llama-3-8B-Instruct or similar) for the cross-family sanity check. Disabled by default (saves ~6 GB VRAM + ~5 min wall clock); set to enable. Evaluate() function now reports `cross_judge_mean` alongside the primary `judge_mean` when enabled.
- **§6.8 end-to-end framing** — explicit caveat that the 26–72× kernel speedup translates to <1% of end-to-end step time at typical §11.7 configurations. Pre-empts "framework is 26× faster" misreading.
- **§3.6 memory algorithm pin-down** — replaces the vague "configurable triggers" with the actual `add_turn` sequence (append → extract entities → extract facts → maybe summarize → decay → trim) plus the retrieval order. Flags semantic-tier embedding-similarity as v1.1 pending work.
- **§8.1 maturity-inversion acknowledgment** — one paragraph explaining why serving is `Stable` while trainers are `Beta`/`Experimental` (the operational layer hardened first against StateSet's commercial deployments).
- **§5.3 GEPO math clarification** — names $E_q[q]$ as the self-normalized importance-sampling denominator, connecting the symbol to particle-filter/population-Monte-Carlo primitives.
- **§11.5 token-soup callout** — promotes the issue #16 failure mode from a table row to a prose callout that reminds readers never to accept a rubric improvement without spot-checking generations.
- **"Skim on first read" labels** on §3.6, §6.7, §6.9, §11.6 — improves navigation for readers using the document as reference.

### Changed

- **`NeuralRewardModel` constructor** in `stateset_agents/training/neural_reward_trainer.py` now takes an `encoder=None` parameter and emits a `WARNING` when the default hash-based smoke-test encoder is used without `suppress_smoke_encoder_warning=True`. The hash encoder cannot learn useful rewards (deterministic pseudo-random per string); the warning quotes the fix and the smoke-test rationale. §4.4 prose updated accordingly: the default path is now explicitly framed as "smoke-test the training loop only."
- **§13.2 prose** — softened from "no other framework ships all three" to "no other framework in the matrix treats all three as first-class concerns in a single deployable package," with concrete examples of which competitors treat which subset as first-class.
- **§1.2 BUSL framing** — trimmed the positioning paragraph to a one-line license declaration; the four-year rationale moves to a footnote.

### Documentation

- All eleven reviewer items above plus the smaller notes (BUSL trim, dense Mermaid diagram in §3 — flagged but not yet split; "skim" labels; principles preface).

### Still pending (Colab-bound, for v1.0/v1.1 first-party results)

- Comparative trainer benchmark run (`whitepaper_v1_comparative_trainers.ipynb` — authored in v0.13.1, awaiting Colab execution).
- vLLM speedup measurement run (`vllm_speedup_benchmark.ipynb` — authored in v0.13.1, awaiting Colab execution).
- Cross-family judge result on the §11.7 generations (notebook supports it in this release; needs a run).

## [0.13.1] - 2026-05-18 — Whitepaper PhD-reviewer round + comparative-trainer notebook + notebook-lint

Closes the v1.0 whitepaper review feedback (A− → unblocking-to-A): comparative trainer notebook, vLLM-speedup notebook, related-work feature matrix, Rust-core measured speedup, §5.7 KL note promotion with TRL line citation, notation consistency, and the issue #16 CI smoke ask.

### Added

- **`notebooks/whitepaper_v1_comparative_trainers.ipynb`** — TRL GRPO vs GSPO vs DAPO on the §11.7 customer-support protocol. Three trainers × three seeds = nine training runs, same baseline, same rubric + LLM-judge eval. Fills the §5.6 comparative trainer table that's been empty since v0.11.6. Skips GEPO (async-oriented) and VAPO (experimental warmup) per the reviewer's recommendation.
- **`notebooks/vllm_speedup_benchmark.ipynb`** — HF generate vs vLLM throughput sweep over `prompt_batch_size ∈ {1, 8, 32, 128}` × `num_generations=4` × `max_tokens=512` on `Qwen2.5-0.5B-Instruct`. Produces the §6.4 "Validated configuration" number with sweep data so readers see the speedup-grows-with-batch claim is verifiable.
- **`scripts/lint_notebooks.py`** + **`make notebook-lint`** — codifies issue #16's lessons into a CI-friendly lint. Checks all ten bundled notebooks for: `asyncio.run()` in Jupyter, `Agent(config=...)` (abstract base), `attn_implementation='flash_attention_2'` (Colab incompatibility). Wired into `make smoke` and `.github/workflows/benchmark-smoke.yml`. Trigger paths simplified to `notebooks/**`.
- **`benchmark_results/whitepaper_v1/rust_vs_python_microbenchmark.json`** — first-party measurement backing §6.8's Rust-core speedup claim. `batch_compute_gae` runs 26-72× faster than the equivalent Python loop across three batch sizes; vectorizable kernels (`compute_group_advantages`) don't see the same benefit. Honestly characterized.

### Fixed

- **`stateset_agents/training/trl_grpo_trainer.py:604`** — mirror of the `gspo_trainer.py` canonical-signature fix from `a2bdde4`. TRL GRPO's reward call was using `compute_reward(trajectory=None, turn=turn, ...)`, only accepted by `MultiObjectiveRewardFunction`. Now uses `compute_reward(turns=[turn], context=...)` so any `RewardFunction` subclass works — required by the new comparative-trainer notebook.
- **`notebooks/quickstart_first_finetune.ipynb`** — three `asyncio.run()` calls in the test-prompt loop converted to top-level await (caught by `notebook-lint`).
- **`notebooks/grade_and_curate_demo.ipynb`** — `asyncio.run()` over scenarios refactored to an `async def` + top-level await.
- **`notebooks/sft_from_curated_demo.ipynb`** — two `asyncio.run()` in inline generate/eval calls converted to top-level await.
- **`.github/workflows/benchmark-smoke.yml`** — notebook validation step now lints all ten bundled notebooks (was six) via `python scripts/lint_notebooks.py` in addition to JSON validity.

### Documentation (PhD reviewer round)

- **§13 Related Work** — replaced the prose paragraph with a 15-capability × 6-framework feature matrix (StateSet Agents vs TRL, OpenRLHF, TRLX, NeMo-Aligner, Verl). New "Distinguishing position" subsection naming the three column-pairs where the framework adds something the others don't.
- **§5.7 "Exact Forward KL"** — promoted from "A Note on KL Divergence" with a three-way comparison (analytical / Schulman k1 / Schulman k3) and TRL line citations (`trl/trainer/ppo_trainer.py:517` and `rloo_trainer.py:436` — both use k1 `0.5 * (logprobs_diff**2).mean()`). Closes the "is that really true of all open implementations?" loophole.
- **§3.1 Tool-call semantics** — new paragraph documenting the four behavior decisions: tool call is part of assistant turn (not separate), credit assignment is per-turn (not per-call), tool errors are reward's responsibility, tool outputs excluded from policy gradient.
- **§6.8 Rust Acceleration Core** — replaces the unbacked performance claim with a measured speedup table: 26-72× on `batch_compute_gae`. Explicit that vectorizable kernels don't see the same benefit; the Rust core is included for recurrence-heavy paths, not as a blanket NumPy replacement.
- **§6.4 vLLM speedup** — adds a "Validated configuration" paragraph naming the specific (model, batch, seq-len, hardware) tuple the integration is validated against, pointing at `vllm_speedup_benchmark.ipynb`.
- **PyPI install-path warning** moved from §"Versioning and Reproducibility" body to a prominent callout immediately after the abstract, with a concrete `pip install git+...` command.
- **Notation consistency** — `ε_R` → `ε_H` throughout §5.2 (GSPO) to match §5.0 notation table and §5.4 (DAPO).
- **§4.4 "Bitter Lesson"** — Sutton citation (new reference [20]) plus a caveat about neural-reward failure modes (reward hacking, judge gaming, calibration drift).
- **§1.2** now cross-references §8.1 Component Maturity so new readers encounter the experimental/beta/stable classification before §8.
- **§1.2 BUSL→Apache 2.0** transition gets a rationale paragraph (no-hosted-derivative protection during the 4-year window, always-permitted use for individual/academic/customer cases, binding contractual conversion in 2029).
- **§2.1 layered-design diagram** upgraded from ASCII to Mermaid (matches the §3 class diagram's quality).
- **§7.4 dashboard** description points readers at `dashboard/README.md` for screenshots and `make dashboard` for the live UI.
- **Appendix C.4 test-count verification** — changed from `pytest --collect-only` (counts but doesn't verify green) to `pytest tests/unit -q --no-header | tail -2` (counts AND confirms pass).

## [0.13.0] - 2026-05-18 — First-party canonical benchmark + KL-anchor safety

The release that lands the v1.0 whitepaper's first canonical first-party benchmark result and codifies a previously-undocumented training foot-gun as a runtime warning. Driven by [issue #16](https://github.com/stateset/stateset-agents/issues/16) — a multi-day debugging session running the bundled benchmarks end-to-end on Colab A100.

### Added

- **`PartialCreditGSM8KReward`** in `stateset_agents.data.gsm8k` — dense-reward variant of `GSM8KReward` with four tiers (`0.0` unparseable / `0.2` parseable / `0.5` within 10% / `1.0` correct). Eliminates the all-zero-groups pathology where the binary reward gives no gradient signal on weak base models. 8 new unit tests covering all four tiers and edge cases.
- **`train_with_gspo` runtime warning** — emits a loud warning when `use_reference_model=False` AND `beta=0.0` AND `len(train_queries) < 100`. This combination destabilizes the policy to gibberish output (see whitepaper §10.5 and `benchmark_results/whitepaper_v1/customer_support_qwen3_5_0_8b_gspo.json`). Warning quotes the safe-default fix and points to the artifact.
- **`notebooks/whitepaper_v1_gsm8k_benchmark_v2.ipynb`** — dense-reward A/B variant of the GSM8K benchmark notebook.
- **`notebooks/customer_support_3seed_judge.ipynb`** — the canonical whitepaper §11.7 publication-gate notebook. Three seeds (42 / 1337 / 2026), both rubric and LLM-judge eval (local `Qwen2.5-1.5B-Instruct` judge — no API key), KL anchor enabled.
- **First-party benchmark artifacts** under `benchmark_results/whitepaper_v1/` (un-ignored via `.gitignore` negation): 5 result files spanning the full headroom spectrum — proof-of-life GSM8K, dense-reward A/B, gibberish KL-anchor-absent failure, near-ceiling stability finding, and the canonical positive result (`customer_support_3seed_judge_qwen25_05b_instruct.json`: judge improvement +0.079, 3-seed agreement).
- **Whitepaper §11.7 "First-Party Reproduction"** — methodology, result table, publication-gate verification, cross-references to §5.1 / §10.5 / §B.1.

### Fixed

- **`gspo_trainer.py:594`** now calls `reward_model.compute_reward(turns=[turn], context=...)` — the canonical `RewardFunction` API documented in §4.1 of the whitepaper. The previous call shape (`trajectory=None, turn=turn, ...`) was only accepted by `MultiObjectiveRewardFunction`'s compatibility shim; every other `RewardFunction` subclass raised `TypeError`. 35/35 `tests/unit/test_gspo_trainer.py` pass; 248/248 `-k "gspo or reward"` sweep passes.
- **`notebooks/customer_support_4h.ipynb`** — eight bug patterns from the GSM8K v1 debug session pre-applied: stale commit pin, missing `transformers/peft/torchao` upgrade, `asyncio.run` in Jupyter, abstract `Agent` base used instead of `MultiTurnAgent`, missing `attn_implementation='sdpa'`, no explicit `train_queries` (rubric context dropped), gradient-checkpointing-incompatible bf16 conv1d backward on Qwen3.5's hybrid layer, and the unsafe default GSPOConfig.
- **Whitepaper §B.1 hyperparameter table** — `beta=0.0` and `use_reference_model=False` defaults carry a "See warning below" flag with an inline warning block.
- **Whitepaper §10.3 "Reward gaming"** — adds the inverse case from the canonical run: rule-based rubric scoring a qualitatively better trained policy *lower* than baseline because the trained model pivots to clarifying questions.
- **Whitepaper §10.5 "Reference-model drift"** — adds documented evidence across the full headroom spectrum (no anchor → gibberish; anchor + ceiling → stable; anchor + headroom → positive transfer).
- **Whitepaper §11.5 failure-modes table** — new row for "trained model emits token soup; rubric score still nonzero".
- **Whitepaper §8.1 maturity matrix** — GSPO trainer entry references the first-party result.
- **Whitepaper anchor** rebased twice this release: `c0dbd68` → `a2bdde4` → `4744c76`. `docs/WHITEPAPER_ERRATA.md` records both re-anchorings.

### Changed

- **`notebooks/customer_support_4h.ipynb`** ships with safe defaults: `use_reference_model=True`, `beta=0.05`, `num_epochs=1`.
- **`.gitignore`** — `benchmark_results/` rule changed from whole-tree-ignore to selective: tracks `whitepaper_v1/**` plus the schema/README/summary files, still ignores ad-hoc local dumps.
- **`docs/WHITEPAPER.md`** front-matter — "experimental results absent" caveat replaced with a concrete summary of §11.7.
- **`notebooks/README.md`** — eight core notebooks (was six in 0.12.0).
- **Whitepaper test-suite count** refreshed in §3.2 and §9 — 2,438 collected (was 1,624 in the v0.11.6 anchor).
- **Whitepaper §9 exception-tuple list** — fixed double-listed `SERIALIZATION_EXCEPTIONS` and removed non-existent `ENVIRONMENT_EXCEPTIONS`; correct names from `stateset_agents/exceptions.py` substituted.

### Documentation

- **`docs/WHITEPAPER_ERRATA.md`** — net-new errata document tracking three anchor rebases (`14c0e65` → `c0dbd68` → `a2bdde4` → `4744c76`) and the v0.11.6 → v0.12.2 corrections.

## [0.12.2] - 2026-05-15 — Docs polish

### Documentation
- **README "Start here" section** now links `CHANGELOG.md` alongside `PLATFORM_TOUR.md`, `COOKBOOK.md`, and `notebooks/README.md` — four discoverability entry points.
- **`docs/PLATFORM_TOUR.md` FAQ** version-upgrade hint bumped from "0.11.7" to "0.12.0" (was stale after the 0.12.x releases).
- **`docs/COOKBOOK.md` Recipe 6** example `stateset-agents version --json` output now shows `version: 0.12.1` (was `0.11.8`). The other fields (git_commit, dependencies) remain illustrative and don't depend on the release.

No functional changes; no test changes; pure docs maintenance from a grep audit of "0.11" references across the repo.

## [0.12.1] - 2026-05-15 — Batch evaluation + sample eval set

### Added
- **`stateset-agents evaluate --scenarios <jsonl> --reward <name>`** — batch-evaluation mode against a saved checkpoint. Scores every row in the JSONL with the named reward (`gsm8k` / `customer_support` / `tool_calling`) and emits a markdown report: mean ± std, pass rate at `--threshold`, per-scenario table with ✅/⚠️/❌ markers. The single-message mode is preserved. Suitable for nightly / PR-blocking evaluation in CI.
- **`examples/sample_eval_set.jsonl`** — 10 bundled customer-support scenarios across 4 intents (refund / technical / billing / general). Drop-in input for the new batch-evaluate mode. See [Cookbook Recipe 5](./docs/COOKBOOK.md).
- **`make smoke-fast`** — runs only the 15 platform-pipeline unit test modules (~60s, ~222 tests, no integration tests, no CLI subprocess overhead). Inner-loop alternative to the full `make smoke`.
- **`notebooks/README.md`** — map of the 6 bundled Colab notebooks with an ASCII flowchart for "which notebook to open for which goal" + a stage/runtime/cost table.

### Fixed
- CHANGELOG 0.12.0 entry claimed "7 self-contained recipes" — Recipe 5 (batch evaluation) brought the count to 8. Updated the 0.12.0 entry to reflect what actually shipped.
- `tests/unit/test_recipe_cli.py::test_list_default_with_no_args` checked for "7 recipes" via a 7-keyword list; updated to 8 with the new `batch` keyword. Also `test_by_short_substring` updated since "debug" now matches Recipe 6 (not 5).

## [0.12.0] - 2026-05-15 — Developer experience + discoverability

### Added
- **`stateset-agents recipe <name>`** — open a cookbook recipe in `$PAGER`. Supports `list`, numeric (`recipe 1`), full slug, and substring match. Sources from `docs/COOKBOOK.md`.
- **`docs/COOKBOOK.md`** — 8 self-contained, copy-paste recipes covering: first fine-tune, iterating from production logs, reproducing a whitepaper number, building a tool-using agent, running a batch evaluation, debugging a stuck reward, handing off to a colleague, running the demos. Sphinx-integrated via `docs/cookbook.rst`. The bundled `examples/sample_eval_set.jsonl` (10 scenarios across 4 intents) gives users a ready-to-run input for the batch-evaluate recipe.
- **`notebooks/quickstart_first_finetune.ipynb`** — Colab mirror of Cookbook Recipe 1. The 6th bundled notebook; covers install → scaffold → train → REPL sanity-check → provenance hand-off.
- **Chat REPL `readline` integration** — up-arrow recalls previous commands; input history persists across sessions at `${XDG_STATE_HOME:-$HOME/.local/state}/stateset-agents/chat_input_history`. Linux/macOS automatic; Windows users can `pip install pyreadline3`.
- **`make smoke-cli`** — verify every CLI subcommand's `--help` loads cleanly. Catches argument-parsing regressions across the now-15 commands.
- **`make health`** — comprehensive platform health check: version + provenance + CHANGELOG entry + CLI smoke + doctor + end-to-end GSM8K pipeline smoke. ~10 seconds, the one-command answer to "is everything wired correctly?"
- **`make demo-all`** — runs `demo` → `demo-curation` → `demo-full-loop` in sequence with section dividers. ~12 seconds total. Screen-share / asciinema friendly.
- **`make new-version VERSION=x.y.z`** — atomically bumps `pyproject.toml` + `stateset_agents/__init__.py` and prepends a CHANGELOG draft with today's date + TODO placeholders.
- **`make changelog-check`** — fails if the version in `pyproject.toml` doesn't have a corresponding `## [x.y.z]` section in `CHANGELOG.md`. Wired into CI so version bumps without CHANGELOG entries fail the build.
- **PLATFORM_TOUR.md** — new tour stop 10 ("Close the loop: curate → SFT → iterate") covering the full curation pipeline; new FAQ entries on `make health`, `recipe`, readline, and the diagnostic surface.
- **CI workflow** — extended to gate `make changelog-check`, `make smoke-cli`, `make health`, all 6 notebooks, and per-template scaffold smoke.

### Fixed
- Two earlier-pass notebooks (`whitepaper_v1_gsm8k_benchmark.ipynb`, `customer_support_4h.ipynb`) called `GSPOTrainer(config, agent, environment)` which would `TypeError` at runtime — the actual constructor requires `(config, model, tokenizer, agent, environment, reward_model)`. Both notebooks now use the canonical `train_with_gspo()` high-level entry point.
- Silent gpt2 fallback in the `/agents/converse` router when `STATESET_DEFAULT_CHECKPOINT` was set but the startup hook failed — now logs a diagnostic warning naming the env var so users see what happened.

## [0.11.8] - 2026-05-14 — Human-in-the-loop curation

### Added
- **`stateset-agents chat`** — interactive in-process REPL against an agent. Flags:
  - `--model` / `--checkpoint` / `--system` / `--max-new-tokens` — agent configuration
  - `--history` — append every turn to JSONL
  - `--replay` — preload from a saved transcript
  - `--grade <reward>` — show live composite-reward scores per turn (✅/⚠️/❌ markers)
- **`stateset-agents tour`** — opens `docs/PLATFORM_TOUR.md` in `$PAGER` or stdout.
- **`AgentConfig.peft_path`** field — loads a pre-trained LoRA adapter from disk. Wired through `MultiTurnAgent.initialize()` via `PeftModel.from_pretrained()`. Closes the `--checkpoint` flow.
- **`AgentService.register_default_checkpoint_agent()`** — at API startup, the lifespan hook reads `STATESET_DEFAULT_CHECKPOINT` / `STATESET_DEFAULT_BASE_MODEL` env vars and registers a usable agent under id `"default"`. Routable via `/agents/default/messages`.
- **`scripts/grade_transcript.py`** — score a JSONL chat transcript with the same reward functions used during training (`gsm8k`, `customer_support`, `tool_calling`). Optional `--output-curated` flag writes high-scoring (prompt, response) tuples to a deduplicated JSONL for use as next-pass training data.
- **`scripts/summarize_graded_batch.py`** — cross-transcript umbrella summary (grand mean, per-transcript table sorted by score).
- **`scripts/prepare_sft_dataset.py`** — converts curated JSONL into three SFT dataset formats (`hf-trainer`, `chat`, `axolotl`). Filters: `--min-score`, repeatable `--source`, `--dedup` by prompt. Stats summary.
- **`scripts/sft_from_curated.py`** — supervised fine-tune on the prepared chat-format JSONL via `transformers.Trainer` + PEFT LoRA. Stub-aware: dry-run on CPU, full training on GPU. Saves a LoRA adapter directory consumable by `AgentConfig.peft_path`. Completes the human-in-the-loop cycle entirely in-framework.
- **`stateset-agents version`** — extended to report git commit hash + key dependency versions (`torch`, `transformers`, `peft`, `trl`, `datasets`, `fastapi`, `vllm`) with ✓/— markers. Single source of truth for provenance.
- **`make full-loop`** umbrella — runs prepare-sft → sft-from-curated as one command, closing the loop in a single invocation.
- **Makefile targets:** `grade-transcript`, `grade-batch` (with optional `CURATED=` / `THRESHOLD=`), `grade-batch-summary`, `prepare-sft`, `sft-from-curated`, `full-loop`, `changelog-check`, `serve-trained`, `release-prep`, `demo`, `demo-curation`, `starter-test`, `smoke`.
- **`docs/PLATFORM_TOUR.md`** — canonical 10-stop guided walk from `pip install` to published v1.0 whitepaper, plus a FAQ answering the 7 questions that come up in practice. Sphinx-integrated via `docs/platform_tour.rst`.
- **Tool-calling benchmark** — `stateset_agents.data.tool_calling_bench` (bundled 8-scenario corpus + 3 sample tools + `ToolCallReward` composite). Completes the three-pillar coverage (math, dialogue, function calling).
- **Tool-calling Colab notebook** — `notebooks/tool_calling_agent_demo.ipynb` parallels the GSM8K and customer-support notebooks.
- **Doctor command enhancements** — surfaces `STATESET_DEFAULT_CHECKPOINT` / `STATESET_DEFAULT_BASE_MODEL`, validates the path exists, warns when base model is unset.
- **`docs/figures/`** convention + the v1.0 release packager (`scripts/release_v1_whitepaper.py`) generates publication figures and the §11.7 markdown snippet in one command.
- **40+ new unit + integration tests.** Total test count now 195+ across the benchmark / scaffolding / chat / grading pipelines, all passing in ~75 s with no GPU.

### Fixed
- Earlier Colab notebooks called `GSPOTrainer(config, agent, environment)` which would `TypeError` at runtime — the actual `GSPOTrainer.__init__` requires `(config, model, tokenizer, agent, environment, reward_model)`. Switched both notebooks to the canonical `train_with_gspo()` high-level entry point.
- The `--checkpoint` CLI flag set `STATESET_DEFAULT_CHECKPOINT` but nothing read it — now wired all the way through the API lifespan + the new `AgentConfig.peft_path` field + the LoRA-from-disk loader.
- Silent gpt2 fallback in the agents-converse router when `STATESET_DEFAULT_CHECKPOINT` was set but the startup hook failed — now logs a diagnostic warning so users see what happened.

### Documentation
- README sections "Chat with your fine-tune locally", "Curate good examples", "Scaffold a fine-tuning project", "Benchmark your fine-tune" document the new developer journey.
- CHANGELOG promoted from `[Unreleased]` to `[0.11.7]` for the empirical-results pipeline, then to `[0.11.8]` for the curation work.

## [0.11.7] - 2026-05-14 — Platform empirical-results pipeline

### Added
- **`stateset_agents.utils.reproducibility`** — central `set_all_seeds()` that seeds Python `random`, NumPy, PyTorch (CPU + CUDA), and Transformers in one call. Singleton `REPRODUCIBILITY_STATE` records what was seeded.
- **`stateset_agents.data.gsm8k`** — GSM8K loader (`load_gsm8k`), gold/predicted answer parsers, and a `GSM8KReward` verifier reward function for single-turn verifiable-reward training.
- **`stateset_agents.data.customer_support_bench`** — bundled 24-scenario multi-turn customer-support corpus across 4 intents (refund / technical / billing / general). `SupportRewardComposite` ships a three-signal rule-based reward (intent ack + brand voice + safety multiplier).
- **`stateset_agents.data.tool_calling_bench`** — bundled 8-scenario tool-calling corpus, 3 sample tools (weather / calculator / search), and a `ToolCallReward` composite (tool selection 40% + parameter correctness 30% + outcome substring 30%). Completes the framework's three-pillar benchmark coverage.
- **`stateset_agents.scaffolding`** — `scaffold_project()` and 4 starter templates (`customer-support`, `gsm8k-math`, `tool-calling-agent`, `minimal`) with full file scaffolds: config.yaml, scenarios.jsonl, reward.py, train.py, eval.py, serve.sh, README.md. `--client-name` flag slugifies into output_dir paths and the W&B project name throughout.
- **CLI subcommands** — `stateset-agents starter <template> <output>` and `stateset-agents benchmark {smoke,phase0,plot,aggregate}`. `stateset-agents serve` gained `--checkpoint` and `--base-model` flags to close the train→serve loop.
- **`scripts/run_phase0_benchmark.py`** — runner with `TaskAdapter` interface (`GSM8KAdapter`, `CustomerSupportAdapter`, `ToolCallingAdapter`). Records baseline + config in schema-compliant JSON per `benchmark_results/SCHEMA.md`.
- **`scripts/aggregate_phase0_results.py`** — reads JSONs, groups by (trainer, model), applies publication gates (3 seeds, σ < 0.10, +0.03 improvement, single commit), emits `summary.md` + `summary.csv` + `passes_gates.json`. Returns exit 1 under `--strict` if any group fails.
- **`scripts/plot_phase0_results.py`** — produces two publication PNGs (pass@1 per trainer with seed-variance error bars, improvement ranking with publication-gate line). Graceful matplotlib-less fallback to a markdown text table.
- **`scripts/release_v1_whitepaper.py`** — one-shot v1.0 release packager. Aggregates → plots → generates the whitepaper §11.7 markdown snippet → copies figures into `docs/figures/` → writes a provenance manifest.
- **Three Colab notebooks** under `notebooks/`:
  - `whitepaper_v1_gsm8k_benchmark.ipynb` — GSM8K + GSPO showcase (~45min on A100)
  - `customer_support_4h.ipynb` — multi-turn customer-support showcase (~3h on A100)
  - `tool_calling_agent_demo.ipynb` — tool-calling showcase (~2h on A100)
- **Makefile targets** — `smoke` (umbrella), `benchmark-smoke`, `benchmark-phase0`, `benchmark-phase0-all`, `benchmark-aggregate`, `benchmark-aggregate-strict`, `benchmark-plot`, `benchmark-publish`, `release-whitepaper-v1`, `release-whitepaper-v1-strict`, `serve-trained`, `starter-test`.
- **CI workflow** — `.github/workflows/benchmark-smoke.yml` runs the full unit-test suite, end-to-end benchmark smoke, scaffold-smoke per template, notebook JSON validation, and `make starter-test` on every relevant PR. ~3 min, no GPU.
- **134 new unit tests** across `test_gsm8k.py`, `test_reproducibility.py`, `test_aggregate_phase0_results.py`, `test_customer_support_bench.py`, `test_tool_calling_bench.py`, `test_task_adapters.py`, `test_scaffolding.py`.

### Fixed
- Earlier Colab notebooks called `GSPOTrainer(config, agent, environment)` but the trainer's actual `__init__` requires `(config, model, tokenizer, agent, environment, reward_model)`. Switched both notebooks to the canonical `train_with_gspo()` high-level entry point.

### Documentation
- README sections "Scaffold a fine-tuning project in 30 seconds" and "Benchmark your fine-tune" document the new developer journey from clone → train → benchmark → serve.
- `benchmark_results/SCHEMA.md` — canonical result JSON schema + publication gates.
- `benchmark_results/README.md` — practitioner-facing pipeline reference.
- Whitepaper appendix C ("Reproducibility Commands") gained the full benchmark-pipeline command list.

## [0.11.6] - 2026-04-20

### Changed
- Expanded the blocking mypy surface from 14 to 35 files, enforcing `disallow_untyped_defs` + `disallow_incomplete_defs` across the highest-traffic core and training modules: `agent`, `agent_backends`, `agent_config`, `trajectory`, `reward_base`, `basic_rewards`, `domain_rewards`, `reward_factories`, `environment`, `environment_base`, `conversation_environment`, `multiturn_agent`, `input_validation`, `memory`, `errors`, `base_trainer`, `trl_grpo_trainer`, `gspo_trainer`, `dapo_trainer`, `gepo_trainer`, and `vapo_trainer`.
- Fixed ~64 strict-mode type violations across the newly gated modules (missing return annotations, untyped `*args`/`**kwargs`, bare `Callable`/`list` generics, `Any | None` declarations for lazy-loaded transformers/peft globals, and `None`-callable narrowing around the optional PEFT loader).

### Fixed
- Realigned deployment artifacts (helm chart, Kubernetes training/production manifests, README, benchmark/autopilot docs) with the released package version so `test_helm_values_use_current_package_version` and `test_selected_kubernetes_and_docs_refs_use_current_package_version` pass on `master`.

## [0.11.1] - 2026-04-02

### Added
- First-class `gemma-4-31b` starter support across packaged training helpers, CLI entrypoints, presets, examples, and regression tests.
- Repo hygiene checks for generated artifacts, backup files, and local tool state, enforced in local workflows and CI.
- A canonical-import regression test to prevent new code from depending on deprecated top-level shim modules.

### Changed
- Bumped the package version to `0.11.1` in package metadata and runtime exports.
- Removed tracked backup files and local autoresearch state from the repository, and tightened contributor guidance around maintenance policy.

## [0.11.0] - 2026-04-02

### Added
- Expo-powered mobile training console under `mobile/` with dashboard, runs, datasets, models, and run-detail screens tailored to RL fine-tuning workflows.
- Mobile data layer that reads live training-lab endpoints when available and falls back to curated mock runs, datasets, models, algorithms, and environments for preview mode.
- Auto-research workflow support for tracking recent training-quality search iterations and benchmark outcomes.

### Changed
- Bumped the package version to `0.11.0` in package metadata and runtime exports.
- Scoped `.gitignore` patterns for `lib/`, `models/`, and `runs/` to the repository root so mobile source files can be tracked and released normally.

## [0.5.0] - 2025-10-19

### Added
- Unit coverage for CLI stub mode, monitoring health checks, and stub backend factories to protect critical workflows.

### Changed
- Extracted stub tokenizer/model scaffolding into `core.agent_backends`, keeping `core.agent` focused on orchestration.
- Modernised test fixtures and README/RELEASE_NOTES to document the new backend module and verification steps.

### Fixed
- Hardened optional dependency guards in performance optimisers and monitoring utilities so imports fail gracefully with actionable guidance.
- Replaced deprecated `asyncio.coroutine` usage in health checks with loop-aware execution that supports sync or async callables.

## [0.4.0] - 2025-10-05

### Added
- End-to-end stub mode surfaced in CLI and examples for fast demos and CI validation.
- Regression coverage ensuring stub responses flow through the computational engine and raw prompt paths.

### Changed
- Standardised all internal and external imports on the `stateset_agents.*` namespace to smooth integration.
- Refreshed documentation and CLI messaging to highlight the stub workflow.

## [0.3.3] - 2025-09-04

## [0.3.4] - 2025-09-22

### Fixed
- Made `stateset_agents.core` proxy resilient to missing optional dependencies by wrapping `async_pool` import in a safe try/except. This prevents import-time failures when `aiohttp` is not installed and allows tests and consumers to import the package without optional extras.

### Changed
- Bumped package version to `0.3.4` in `pyproject.toml` and `stateset_agents/__init__.py` to keep versions in sync.

### Notes
- No API changes; this release improves import robustness and packaging hygiene only.

### Changed
- Refactored `training` package imports to absolute paths (`core.*`, `utils.*`, `rewards.*`) so `import training` works directly from installed packages
- CI publish workflow hardened: idempotent uploads (`twine upload --skip-existing`), resilient tag creation, and Docker/docs jobs also run on GitHub release events

### Fixed
- Resolved import errors when using `training` modules from PyPI install
- Ensured packaged distribution includes all required top-level modules used by `stateset_agents`

### Publishing
- Released `stateset-agents==0.3.3` to PyPI

## [0.3.2] - 2025-09-04

### Added
- Packaged top-level implementation modules with the library: `core`, `rewards`, `utils`, `api`, `environments`, and a lightweight `grpo_agent_framework` proxy

### Fixed
- Added proxy module for `stateset_agents.core` to forward to top-level `core` package to avoid duplication while enabling stable imports

### Publishing
- Released `stateset-agents==0.3.2` to TestPyPI and PyPI

## [0.3.1] - 2024-12-XX

### Added
- **Code Quality Tools**: Comprehensive code quality setup with Black, isort, Ruff, mypy, and pre-commit hooks
- **CI/CD Pipeline**: GitHub Actions workflows for automated testing, linting, security scanning, and publishing
- **Development Tools**: Makefile with common development tasks and comprehensive pyproject.toml configuration
- **Documentation**: Code of Conduct and Contributing guidelines for community standards
- **Security**: Bandit security linting and dependency vulnerability scanning
- **Benchmarking**: Automated performance benchmarking with historical tracking
- **Dependency Management**: Dependabot configuration for automated dependency updates

### Changed
- **Package Consistency**: Standardized all imports to use `stateset_agents` namespace consistently
- **Documentation**: Updated all examples and documentation to use correct import paths
- **Testing**: Enhanced pytest configuration with coverage thresholds and better markers
- **Project Structure**: Improved .gitignore and development file organization

### Fixed
- Import inconsistencies between `grpo_agent_framework` and `stateset_agents` packages
- Missing development dependencies and tools
- Inconsistent code formatting and style

### Developer Experience
- Added comprehensive Makefile for common tasks (`make help` to see all commands)
- Pre-commit hooks for automated code quality checks
- Enhanced testing with coverage reporting and multiple test categories
- Improved documentation with clear contribution guidelines

## [0.3.0] - 2024-01-XX

### Added
- **Enhanced Error Handling & Resilience**
  - Comprehensive exception hierarchy for training, model, data, network, and resource errors
  - Configurable async retry with exponential backoff and jitter
  - Circuit breaker pattern for automatic failure detection and recovery
  - Rich error context with stack traces, categories, and recovery suggestions

- **Performance Optimization**
  - Real-time memory monitoring with automatic cleanup and optimization
  - Dynamic batch sizing based on resource availability
  - PyTorch 2.0 compilation support for faster inference
  - Mixed precision training with automated FP16/BF16 optimization

- **Type Safety & Validation**
  - Runtime type checking for all framework components
  - Type-safe configuration with detailed error reporting
  - Reliable serialization/deserialization with type preservation
  - Clear protocol interfaces for extensible components

- **Advanced Async Resource Management**
  - High-performance async resource pools with health checking
  - Sophisticated async task scheduling with resource limits
  - Automatic resource cleanup and scaling
  - Real-time monitoring of resource utilization

- **Production Monitoring**
  - Comprehensive performance tracking and reporting
  - Automated system health monitoring and alerting
  - Dynamic optimization recommendations
  - Advanced debugging and profiling capabilities

### Changed
- Enhanced production-ready configuration and deployment
- Improved async resource management patterns
- Better error handling and recovery mechanisms
- More comprehensive monitoring and observability

## [0.2.0] - 2023-11-XX

### Added
- Multi-turn conversation support with trajectory tracking
- Domain-specific reward functions (Customer Service, Technical Support, Sales)
- Neural reward models that learn from trajectory data
- Distributed training capabilities
- Advanced data processing pipeline
- TRL GRPO integration for large model fine-tuning

### Changed
- Enhanced training infrastructure with better stability
- Improved reward modeling system
- Better environment abstractions

## [0.1.0] - 2023-08-XX

### Added
- Initial GRPO training framework
- Basic agent and environment classes
- Core reward functions
- Simple training pipeline
- Documentation and examples

### Changed
- Initial release with core functionality
