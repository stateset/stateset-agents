# Kimi-K3 First-Class Starter — Design

**Date:** 2026-07-16
**Status:** Approved
**Pattern precedent:** `kimi_k2_6_starter.py` (commit 1ff42de), GLM-5.2 provisional-spec precedent (commit 387dcce)

## Goal

Add Moonshot AI's Kimi-K3 as a first-class starter in the stack, mirroring the
Kimi-K2.6 surface: packaged starter module, CLI command, example scripts, docs,
and tests.

## Context and constraints

- Kimi K3 launched on Moonshot's product surface on 2026-07-16 (~2.5T-param
  MoE, 1M+ token context), but **no HuggingFace weights, model card, or license
  are published yet**.
- Model ID is therefore provisional: `moonshotai/Kimi-K3`, following Moonshot's
  org naming. Single supported variant.
- Profile presets are a straight mirror of Kimi-K2.6's numbers (QLoRA 4-bit
  budgets tuned for a ~1T MoE). These are training-side VRAM budgets, not model
  capability claims; they get revisited when the HF card lands. The module
  docstring and docs state that specs are provisional.

## Components

### 1. Starter module — `stateset_agents/training/kimi_k3_starter.py`

Exact structural mirror of `kimi_k2_6_starter.py`:

- Constants: `KIMI_K3_BASE_MODEL = "moonshotai/Kimi-K3"`,
  `KIMI_K3_SUPPORTED_VARIANTS` (single entry), `KIMI_K3_TASK_CHOICES`
  (`customer_service`, `technical_support`, `sales`, `conversational`),
  `KIMI_K3_STARTER_PROFILE_CHOICES` (`balanced`, `memory`, `quality`) with
  descriptions, `KIMI_K3_DEFAULT_OUTPUT_DIR = "./outputs/kimi_k3_gspo"`,
  `KIMI_K3_LORA_TARGET_MODULES` (same 7 projection modules),
  `KIMI_K3_CONFIG_SUFFIXES`.
- `KimiK3Config` dataclass with the same fields and defaults as `KimiK26Config`.
- Function surface: `get_kimi_k3_system_prompt`, `get_kimi_k3_profile_overrides`,
  `get_kimi_k3_profile_description`, `summarize_kimi_k3_config`,
  `describe_kimi_k3_starter_profiles`, `get_kimi_k3_config`,
  `create_kimi_k3_preview`, `run_kimi_k3_config`, `write_kimi_k3_config_file`,
  `load_kimi_k3_config_file`, `finetune_kimi_k3`.
- W&B tags: `["kimi-k3", "gspo", <task>]`.
- System prompts keep the "You are Kimi, an AI assistant created by Moonshot
  AI." intro.

### 2. Wiring

- Lazy-export block in `stateset_agents/training/__init__.py`, same shape as
  the K2.6 and GLM-5.2 blocks.
- `@app.command("kimi-k3")` in `stateset_agents/cli_train.py` (post-split home
  of `kimi-k2-6`), supporting `--list-profiles`, `--write-config`, `--config`,
  `--starter-profile {balanced,memory,quality}`, `--task`, `--json-output`,
  `--dry-run/--no-dry-run`.

### 3. Examples

- `examples/finetune_kimi_k3_gspo.py` (mirror of `finetune_kimi_k2_6_gspo.py`)
- `examples/kimi_k3_config.py` (mirror of `kimi_k2_6_config.py`)
- Entry in `examples/README.md`.

### 4. Docs

- `docs/kimi_k3_starter.rst`, wired into `docs/index.rst` toctree and
  `docs/examples.rst`.
- New `kimi-k3` section in `docs/CLI_REFERENCE.md`.
- First-class row in `docs/SUPPORTED_MODELS.md`, noting "weights pending HF
  release".
- README mention alongside the other starters.
- CHANGELOG entry under Unreleased. **No package version bump** (GLM-5.2
  precedent — release commits handle bumps separately).

### 5. Tests

- `tests/unit/test_kimi_k3_config.py`: config resolution, profile overrides,
  validation warnings, JSON/YAML config round-trip, error paths (bad profile,
  bad task, bad suffix). Mirrors `test_kimi_k2_6_config.py`.
- `tests/unit/test_kimi_k3_module_exports.py`: lazy-import surface.
- `kimi-k3` CLI tests in `tests/unit/test_cli.py`, mirroring the K2.6 CLI test
  additions (list-profiles, write-config, dry-run, JSON output).
- All tests run on the stub backend; no model weights required.

## Out of scope

- Kubernetes/vLLM/Helm deployment manifests, render script, and hosting plan
  doc — deferred until real weights, license, and GPU sizing exist.
- FP8 or quantized variant aliases.
- Any changes to K2.5/K2.6 surfaces.

## Success criteria

- `from stateset_agents.training import kimi_k3_starter` and the lazy exports
  work.
- `stateset-agents kimi-k3 --list-profiles` and `--dry-run` succeed without
  downloading weights.
- New unit tests pass; existing suite stays green.
- Docs build includes `kimi_k3_starter.rst` without Sphinx warnings.
