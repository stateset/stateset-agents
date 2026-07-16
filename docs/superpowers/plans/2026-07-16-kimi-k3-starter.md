# Kimi-K3 First-Class Starter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Moonshot AI's Kimi-K3 as a first-class starter: packaged module, `stateset-agents kimi-k3` CLI command, `init --preset kimi-k3`, examples, docs, and tests — a structural mirror of the existing Kimi-K2.6 starter.

**Architecture:** Every artifact is a mechanical mirror of its Kimi-K2.6 counterpart, produced by copying the K2.6 source and applying a fixed token-rename map, then adding provisional-release notes where user-facing. TDD per task: generate the mirrored test file first (it fails because the K3 surface doesn't exist), then generate the implementation, then verify.

**Tech Stack:** Python 3.10, Typer CLI, pytest, Sphinx/RST docs. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-16-kimi-k3-starter-design.md`

## Global Constraints

- Model ID is exactly `moonshotai/Kimi-K3` — provisional; HF weights/card/license unpublished as of 2026-07-16. Single supported variant.
- Profile presets are a straight mirror of Kimi-K2.6's numbers. Do NOT invent different batch/context/LR values.
- No package version bump. CHANGELOG entry goes under `## [Unreleased]`.
- No deployment manifests, no FP8/quantized variant aliases, no edits to any Kimi-K2.5 or Kimi-K2.6 file (`git diff` must show K2.5/K2.6 files untouched).
- All work happens on branch `feat/kimi-k3-starter` (already created; spec is committed there).
- **Canonical rename map** (applied in this order everywhere; patterns are disjoint so order is safe):
  1. `KIMI_K26` → `KIMI_K3`
  2. `KimiK26` → `KimiK3`
  3. `kimi_k2_6` → `kimi_k3`
  4. `Kimi-K2.6` → `Kimi-K3`
  5. `kimi-k2.6` → `kimi-k3`
  6. `kimi-k2-6` → `kimi-k3`
- **Residual-token check** after every generation step: `grep -nE 'K26|k2_6|[Kk]2\.6|k2-6' <generated files>` must print nothing. If it prints anything, hand-fix those lines to the K3 equivalent before proceeding.
- Run tests with `python -m pytest <path> -v` from the repo root (`/home/dom/stateset-agents`).

---

### Task 1: Starter module, example scripts, and config tests

**Files:**
- Create: `stateset_agents/training/kimi_k3_starter.py` (from `stateset_agents/training/kimi_k2_6_starter.py`, 595 lines)
- Create: `examples/finetune_kimi_k3_gspo.py` (from `examples/finetune_kimi_k2_6_gspo.py`, 219 lines)
- Create: `examples/kimi_k3_config.py` (from `examples/kimi_k2_6_config.py`, 66 lines)
- Test: `tests/unit/test_kimi_k3_config.py` (from `tests/unit/test_kimi_k2_6_config.py`, 294 lines)

**Interfaces:**
- Consumes: `stateset_agents.core.agent.AgentConfig`, `stateset_agents.training.config.TrainingConfig` (existing, unchanged).
- Produces (later tasks import these by exact name from `stateset_agents.training.kimi_k3_starter`): constants `KIMI_K3_BASE_MODEL = "moonshotai/Kimi-K3"`, `KIMI_K3_SUPPORTED_VARIANTS`, `KIMI_K3_TASK_CHOICES`, `KIMI_K3_STARTER_PROFILE_CHOICES`, `KIMI_K3_STARTER_PROFILE_DESCRIPTIONS`, `KIMI_K3_DEFAULT_OUTPUT_DIR = "./outputs/kimi_k3_gspo"`, `KIMI_K3_LORA_TARGET_MODULES`, `KIMI_K3_CONFIG_SUFFIXES`; dataclass `KimiK3Config`; functions `get_kimi_k3_system_prompt(task)`, `get_kimi_k3_profile_overrides(starter_profile)`, `get_kimi_k3_profile_description(starter_profile)`, `summarize_kimi_k3_config(config)`, `describe_kimi_k3_starter_profiles(task, model_name)`, `get_kimi_k3_config(...) -> KimiK3Config`, `get_kimi_k3_gspo_config`, `get_kimi_k3_gspo_overrides`, `create_kimi_k3_agent_config`, `create_kimi_k3_preview`, `validate_kimi_k3_config`, `run_kimi_k3_config`, `write_kimi_k3_config_file`, `load_kimi_k3_config_file`, `finetune_kimi_k3` — identical signatures to their `*_kimi_k2_6_*` counterparts.

- [ ] **Step 1: Generate the failing test file**

```bash
sed -e 's/KIMI_K26/KIMI_K3/g' -e 's/KimiK26/KimiK3/g' -e 's/kimi_k2_6/kimi_k3/g' \
    -e 's/Kimi-K2\.6/Kimi-K3/g' -e 's/kimi-k2\.6/kimi-k3/g' -e 's/kimi-k2-6/kimi-k3/g' \
    tests/unit/test_kimi_k2_6_config.py > tests/unit/test_kimi_k3_config.py
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_kimi_k3_config.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'stateset_agents.training.kimi_k3_starter'`

- [ ] **Step 3: Generate the starter module and both example scripts**

```bash
sed -e 's/KIMI_K26/KIMI_K3/g' -e 's/KimiK26/KimiK3/g' -e 's/kimi_k2_6/kimi_k3/g' \
    -e 's/Kimi-K2\.6/Kimi-K3/g' -e 's/kimi-k2\.6/kimi-k3/g' -e 's/kimi-k2-6/kimi-k3/g' \
    stateset_agents/training/kimi_k2_6_starter.py > stateset_agents/training/kimi_k3_starter.py
sed -e 's/KIMI_K26/KIMI_K3/g' -e 's/KimiK26/KimiK3/g' -e 's/kimi_k2_6/kimi_k3/g' \
    -e 's/Kimi-K2\.6/Kimi-K3/g' -e 's/kimi-k2\.6/kimi-k3/g' -e 's/kimi-k2-6/kimi-k3/g' \
    examples/finetune_kimi_k2_6_gspo.py > examples/finetune_kimi_k3_gspo.py
sed -e 's/KIMI_K26/KIMI_K3/g' -e 's/KimiK26/KimiK3/g' -e 's/kimi_k2_6/kimi_k3/g' \
    -e 's/Kimi-K2\.6/Kimi-K3/g' -e 's/kimi-k2\.6/kimi-k3/g' -e 's/kimi-k2-6/kimi-k3/g' \
    examples/kimi_k2_6_config.py > examples/kimi_k3_config.py
```

- [ ] **Step 4: Add the provisional-release note to the module docstring**

In `stateset_agents/training/kimi_k3_starter.py`, replace the first line:

```python
"""Packaged Kimi-K3 GSPO starter helpers."""
```

with:

```python
"""Packaged Kimi-K3 GSPO starter helpers.

Kimi K3 launched on Moonshot's product surface on 2026-07-16 (~2.5T-param MoE,
1M+ token native context per press coverage), but HuggingFace weights, model
card, and license are not yet published. ``KIMI_K3_BASE_MODEL`` and the profile
presets below are provisional mirrors of the Kimi-K2.6 starter pending the
official release.
"""
```

- [ ] **Step 5: Residual-token check**

Run: `grep -nE 'K26|k2_6|[Kk]2\.6|k2-6' stateset_agents/training/kimi_k3_starter.py examples/finetune_kimi_k3_gspo.py examples/kimi_k3_config.py tests/unit/test_kimi_k3_config.py`
Expected: no output. If any line prints, hand-edit it to the K3 equivalent.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_kimi_k3_config.py -v`
Expected: PASS — same test count as `python -m pytest tests/unit/test_kimi_k2_6_config.py --collect-only -q` reports for K2.6, zero failures. (The subprocess-based tests in this file invoke `examples/finetune_kimi_k3_gspo.py`, which is why the examples are created in this task.)

- [ ] **Step 7: Regression check on the K2.6 twin**

Run: `python -m pytest tests/unit/test_kimi_k2_6_config.py -q`
Expected: PASS (proves the K2.6 surface is untouched).

- [ ] **Step 8: Commit**

```bash
git add stateset_agents/training/kimi_k3_starter.py examples/finetune_kimi_k3_gspo.py examples/kimi_k3_config.py tests/unit/test_kimi_k3_config.py
git commit -m "feat: add Kimi-K3 starter module and examples (provisional specs)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Lazy exports from `stateset_agents.training`

**Files:**
- Modify: `stateset_agents/training/__init__.py` (lazy-import map block after the `# Kimi-K2.6 starter path` block at ~line 153-177; `__all__` block after `"write_kimi_k2_6_config_file",` at ~line 450)
- Test: `tests/unit/test_kimi_k3_module_exports.py` (from `tests/unit/test_kimi_k2_6_module_exports.py`, 48 lines)

**Interfaces:**
- Consumes: `stateset_agents.training.kimi_k3_starter` (Task 1) — all 24 public names listed in Task 1's Produces block.
- Produces: the same 24 names lazily importable from `stateset_agents.training` (e.g., `from stateset_agents.training import KIMI_K3_BASE_MODEL, KimiK3Config, get_kimi_k3_config`).

- [ ] **Step 1: Generate the failing test file**

```bash
sed -e 's/KIMI_K26/KIMI_K3/g' -e 's/KimiK26/KimiK3/g' -e 's/kimi_k2_6/kimi_k3/g' \
    -e 's/Kimi-K2\.6/Kimi-K3/g' -e 's/kimi-k2\.6/kimi-k3/g' -e 's/kimi-k2-6/kimi-k3/g' \
    tests/unit/test_kimi_k2_6_module_exports.py > tests/unit/test_kimi_k3_module_exports.py
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_kimi_k3_module_exports.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'KIMI_K3_BASE_MODEL' from 'stateset_agents.training'`

- [ ] **Step 3: Insert the renamed lazy-map and `__all__` blocks**

```bash
python3 - <<'EOF'
from pathlib import Path

p = Path("stateset_agents/training/__init__.py")
src = p.read_text()

def ren(s):
    for a, b in [
        ("KIMI_K26", "KIMI_K3"), ("KimiK26", "KimiK3"), ("kimi_k2_6", "kimi_k3"),
        ("Kimi-K2.6", "Kimi-K3"), ("kimi-k2.6", "kimi-k3"), ("kimi-k2-6", "kimi-k3"),
    ]:
        s = s.replace(a, b)
    return s

# 1) Lazy-import map: copy the K2.6 block (comment header through last entry)
#    and insert the renamed copy directly after it, before "# Serving artifacts".
map_start = src.index("    # Kimi-K2.6 starter path")
map_end = src.index("    # Serving artifacts")
src = src[:map_end] + ren(src[map_start:map_end]) + src[map_end:]

# 2) __all__: copy the K26 entry block and insert the renamed copy after it.
blk_start = src.index('    "KIMI_K26_BASE_MODEL",\n', src.index("__all__"))
end_marker = '    "write_kimi_k2_6_config_file",\n'
blk_end = src.index(end_marker, blk_start) + len(end_marker)
src = src[:blk_end] + ren(src[blk_start:blk_end]) + src[blk_end:]

p.write_text(src)
EOF
```

Expected resulting shape (spot-check): the lazy map now has a `    # Kimi-K3 starter path` comment followed by 24 entries mapping e.g. `"KIMI_K3_BASE_MODEL": (f"{__name__}.kimi_k3_starter", "KIMI_K3_BASE_MODEL")`, and `__all__` has the 24 matching string entries after `"write_kimi_k2_6_config_file",`.

- [ ] **Step 4: Run test to verify it passes, plus twins**

Run: `python -m pytest tests/unit/test_kimi_k3_module_exports.py tests/unit/test_kimi_k2_6_module_exports.py -v`
Expected: PASS (both files).

- [ ] **Step 5: Commit**

```bash
git add stateset_agents/training/__init__.py tests/unit/test_kimi_k3_module_exports.py
git commit -m "feat: export Kimi-K3 starter from stateset_agents.training

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `kimi-k3` CLI command and `init --preset kimi-k3`

**Files:**
- Modify: `stateset_agents/cli_train.py` (insert new command between `kimi_k2_6` at lines 495-762 and `@app.command("gemma-4-31b")` at line 763)
- Modify: `stateset_agents/cli.py` (init preset: help strings at ~726 and ~937, validation set at ~744, error messages at ~745-752, new `elif` branch after the `kimi-k2-6` branch at ~840-874)
- Test: `tests/unit/test_cli.py` (mirror the `test_cli_kimi_k2_6_*` block at lines 346-485 and the `test_cli_init_kimi_preset_*` block at lines 942-1012)

**Interfaces:**
- Consumes: direct imports `from stateset_agents.training.kimi_k3_starter import (KIMI_K3_BASE_MODEL, KIMI_K3_STARTER_PROFILE_CHOICES, KIMI_K3_TASK_CHOICES, create_kimi_k3_preview, describe_kimi_k3_starter_profiles, get_kimi_k3_config, load_kimi_k3_config_file, run_kimi_k3_config, write_kimi_k3_config_file)` (Task 1; the lazy exports from Task 2 are NOT used by the CLI).
- Produces: Typer commands `stateset-agents kimi-k3` (same flags as `kimi-k2-6`: `--config`, `--task`, `--starter-profile`, `--list-profiles`, `--model`, `--use-lora/--no-lora`, `--use-4bit`, `--use-8bit`, `--output-dir`, `--iterations`, `--wandb`, `--wandb-project`, `--write-config`, `--dry-run/--no-dry-run`, `--json-output`) and `stateset-agents init --preset kimi-k3`.

- [ ] **Step 1: Generate the failing CLI tests**

```bash
python3 - <<'EOF'
from pathlib import Path

p = Path("tests/unit/test_cli.py")
src = p.read_text()

def ren(s):
    for a, b in [
        ("KIMI_K26", "KIMI_K3"), ("KimiK26", "KimiK3"), ("kimi_k2_6", "kimi_k3"),
        ("Kimi-K2.6", "Kimi-K3"), ("kimi-k2.6", "kimi-k3"), ("kimi-k2-6", "kimi-k3"),
        ("test_cli_init_kimi_preset", "test_cli_init_kimi_k3_preset"),
    ]:
        s = s.replace(a, b)
    return s

# kimi-k3 command tests (mirror of the kimi-k2-6 command test block)
start = src.index("def test_cli_kimi_k2_6_dry_run_json")
end = src.index("def test_cli_validate_config_command_success")
src = src[:end] + ren(src[start:end]) + src[end:]

# init --preset kimi-k3 tests (mirror of the init kimi preset test block)
start = src.index("def test_cli_init_kimi_preset_json")
end = src.index("def test_cli_init_gemma_preset_json")
src = src[:end] + ren(src[start:end]) + src[end:]

p.write_text(src)
EOF
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_cli.py -k "kimi_k3" -v`
Expected: FAIL — the `kimi-k3` command tests exit with code 2 (Typer: no such command), and `init --preset kimi-k3` tests exit with code 2 (unsupported preset), where the tests assert exit code 0.

- [ ] **Step 3: Insert the `kimi-k3` command into `cli_train.py`**

```bash
python3 - <<'EOF'
from pathlib import Path

p = Path("stateset_agents/cli_train.py")
src = p.read_text()

def ren(s):
    for a, b in [
        ("KIMI_K26", "KIMI_K3"), ("KimiK26", "KimiK3"), ("kimi_k2_6", "kimi_k3"),
        ("Kimi-K2.6", "Kimi-K3"), ("kimi-k2.6", "kimi-k3"), ("kimi-k2-6", "kimi-k3"),
    ]:
        s = s.replace(a, b)
    return s

start = src.index('@app.command("kimi-k2-6")')
end = src.index('@app.command("gemma-4-31b")')
src = src[:end] + ren(src[start:end]) + src[end:]
p.write_text(src)
EOF
```

Expected resulting shape: a new `@app.command("kimi-k3")` / `def kimi_k3(...)` function sits between `def kimi_k2_6` and `@app.command("gemma-4-31b")`, byte-identical to `kimi_k2_6` modulo the rename map.

- [ ] **Step 4: Add the `kimi-k3` init preset to `cli.py`**

Four edits:

4a. Both `--preset` help strings (occurs twice — `init` and its alias; replace both):

```python
        help="Starter preset: default, qwen3-5-0-8b, kimi-k2-6, or gemma-4-31b",
```
becomes
```python
        help="Starter preset: default, qwen3-5-0-8b, kimi-k2-6, kimi-k3, or gemma-4-31b",
```

4b. Validation set and its error message:

```python
    if preset not in {"default", "qwen3-5-0-8b", "kimi-k2-6", "gemma-4-31b"}:
        _echo(
            "Unsupported preset. Use one of: default, qwen3-5-0-8b, kimi-k2-6, gemma-4-31b."
        )
```
becomes
```python
    if preset not in {"default", "qwen3-5-0-8b", "kimi-k2-6", "kimi-k3", "gemma-4-31b"}:
        _echo(
            "Unsupported preset. Use one of: default, qwen3-5-0-8b, kimi-k2-6, kimi-k3, gemma-4-31b."
        )
```

4c. The `--starter-profile` guard message:

```python
            "`--starter-profile` only applies to --preset qwen3-5-0-8b, kimi-k2-6, or gemma-4-31b."
```
becomes
```python
            "`--starter-profile` only applies to --preset qwen3-5-0-8b, kimi-k2-6, kimi-k3, or gemma-4-31b."
```

4d. Insert the renamed preset branch after the `kimi-k2-6` branch:

```bash
python3 - <<'EOF'
from pathlib import Path

p = Path("stateset_agents/cli.py")
src = p.read_text()

def ren(s):
    for a, b in [
        ("KIMI_K26", "KIMI_K3"), ("KimiK26", "KimiK3"), ("kimi_k2_6", "kimi_k3"),
        ("Kimi-K2.6", "Kimi-K3"), ("kimi-k2.6", "kimi-k3"), ("kimi-k2-6", "kimi-k3"),
    ]:
        s = s.replace(a, b)
    return s

start = src.index('    elif preset == "kimi-k2-6":')
end = src.index("    else:", start)
src = src[:end] + ren(src[start:end]) + src[end:]
p.write_text(src)
EOF
```

Expected resulting branch (verify it reads exactly like this):

```python
    elif preset == "kimi-k3":
        try:
            from stateset_agents.training.kimi_k3_starter import (
                KIMI_K3_STARTER_PROFILE_CHOICES,
                KIMI_K3_TASK_CHOICES,
                get_kimi_k3_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Kimi-K3 starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in KIMI_K3_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(KIMI_K3_TASK_CHOICES)}.")
            raise typer.Exit(code=2)
        if starter_profile not in KIMI_K3_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(KIMI_K3_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_kimi_k3_config(task=task, starter_profile=starter_profile).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
```

- [ ] **Step 5: Run tests to verify they pass, plus twins**

Run: `python -m pytest tests/unit/test_cli.py -k "kimi" -v`
Expected: PASS — all `kimi_k3` tests plus all pre-existing `kimi_k2_6` / `init_kimi_preset` tests.

- [ ] **Step 6: CLI smoke check**

Run: `stateset-agents kimi-k3 --list-profiles --json-output | head -5`
Expected: JSON payload beginning with `{"model_name": "moonshotai/Kimi-K3", ...` listing `balanced`, `memory`, `quality`.

- [ ] **Step 7: Commit**

```bash
git add stateset_agents/cli_train.py stateset_agents/cli.py tests/unit/test_cli.py
git commit -m "feat: add kimi-k3 CLI command and init preset

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Docs, README, CHANGELOG

**Files:**
- Create: `docs/kimi_k3_starter.rst` (from `docs/kimi_k2_6_starter.rst`, 84 lines)
- Modify: `docs/index.rst` (toctree, after `   kimi_k2_6_starter` at line 71)
- Modify: `docs/examples.rst` (after the `kimi_k2_6_starter` line at line 11)
- Modify: `docs/CLI_REFERENCE.md` (new section after `### \`stateset-agents kimi-k2-6\`` at lines 69-101; fix stale preset list at line ~176)
- Modify: `docs/SUPPORTED_MODELS.md` (first-class table, after the GLM 5.2 row at line 24)
- Modify: `README.md` (new starter section after the Kimi-K2.6 section ending at line ~177; supported-models sentence at line 241)
- Modify: `examples/README.md` (Kimi Models section at lines 224-236)
- Modify: `CHANGELOG.md` (new `### Added` section under `## [Unreleased]` at line 9)

**Interfaces:**
- Consumes: command name `kimi-k3`, preset name `kimi-k3`, module path `stateset_agents/training/kimi_k3_starter.py`, example paths from Tasks 1-3. No code changes.
- Produces: documentation only.

- [ ] **Step 1: Generate `docs/kimi_k3_starter.rst`**

```bash
sed -e 's/KIMI_K26/KIMI_K3/g' -e 's/KimiK26/KimiK3/g' -e 's/kimi_k2_6/kimi_k3/g' \
    -e 's/Kimi-K2\.6/Kimi-K3/g' -e 's/kimi-k2\.6/kimi-k3/g' -e 's/kimi-k2-6/kimi-k3/g' \
    docs/kimi_k2_6_starter.rst > docs/kimi_k3_starter.rst
```

Then two hand edits in `docs/kimi_k3_starter.rst`:

1. Fix the title underline (title shrank from 22 to 20 chars):

```rst
Kimi-K3 Starter Path
====================
```

2. Insert a provisional note after the two intro lines. Replace:

```rst
Use this starter when you want the fastest path to a first GSPO post-training run for ``moonshotai/Kimi-K3``.
The recommended checkpoint for post-training is ``moonshotai/Kimi-K3``.
```

with:

```rst
Use this starter when you want the fastest path to a first GSPO post-training run for ``moonshotai/Kimi-K3``.
The recommended checkpoint for post-training is ``moonshotai/Kimi-K3``.

.. note::

   Kimi K3 launched on Moonshot's product surface on 2026-07-16, but HuggingFace
   weights, model card, and license are not yet published. The
   ``moonshotai/Kimi-K3`` model ID and the profile presets in this starter are
   provisional mirrors of the Kimi-K2.6 starter pending the official release.
```

- [ ] **Step 2: Wire into `docs/index.rst` and `docs/examples.rst`**

In `docs/index.rst`, replace:

```rst
   kimi_k2_6_starter
```

with:

```rst
   kimi_k2_6_starter
   kimi_k3_starter
```

In `docs/examples.rst`, replace:

```rst
- :doc:`kimi_k2_6_starter`: built-docs quick start for post-training ``moonshotai/Kimi-K2.6``.
```

with:

```rst
- :doc:`kimi_k2_6_starter`: built-docs quick start for post-training ``moonshotai/Kimi-K2.6``.
- :doc:`kimi_k3_starter`: built-docs quick start for post-training ``moonshotai/Kimi-K3`` (weights pending HF release).
```

- [ ] **Step 3: Add the `kimi-k3` section to `docs/CLI_REFERENCE.md`**

```bash
python3 - <<'EOF'
from pathlib import Path

p = Path("docs/CLI_REFERENCE.md")
src = p.read_text()

def ren(s):
    for a, b in [
        ("KIMI_K26", "KIMI_K3"), ("KimiK26", "KimiK3"), ("kimi_k2_6", "kimi_k3"),
        ("Kimi-K2.6", "Kimi-K3"), ("kimi-k2.6", "kimi-k3"), ("kimi-k2-6", "kimi-k3"),
    ]:
        s = s.replace(a, b)
    return s

start = src.index("### `stateset-agents kimi-k2-6`")
end = src.index("### `stateset-agents validate-config`")
src = src[:end] + ren(src[start:end]) + src[end:]
p.write_text(src)
EOF
```

Then in the NEW `### \`stateset-agents kimi-k3\`` section, replace:

```markdown
Preview or run the dedicated starter path for `moonshotai/Kimi-K3`.
The command defaults to a dry-run so you can inspect the resolved config before loading a model.
```

with:

```markdown
Preview or run the dedicated starter path for `moonshotai/Kimi-K3`.
The command defaults to a dry-run so you can inspect the resolved config before loading a model.
Note: the `moonshotai/Kimi-K3` ID is provisional — HF weights are not yet published (as of 2026-07-16).
```

And fix the stale `init` preset list. Replace:

```markdown
- `--preset [default|qwen3-5-0-8b]`: Starter config preset.
```

with:

```markdown
- `--preset [default|qwen3-5-0-8b|kimi-k2-6|kimi-k3|gemma-4-31b]`: Starter config preset.
```

- [ ] **Step 4: Add the `docs/SUPPORTED_MODELS.md` row**

In the first-class starters table, after the GLM 5.2 row, add:

```markdown
| **Kimi-K3** | `moonshotai/Kimi-K3` *(provisional — HF weights pending as of 2026-07-16)* | `stateset-agents kimi-k3` | `examples/finetune_kimi_k3_gspo.py` | Mirrors K2.6 presets pending official specs |
```

- [ ] **Step 5: Update `README.md`**

5a. After the Kimi-K2.6 starter section (its closing line is `Use \`--list-profiles\` when you want to compare the built-in \`balanced\`, \`memory\`, and \`quality\` presets before saving or running one.` followed by a blank line and `### Gemma 4 31B starter path`), insert a new section before `### Gemma 4 31B starter path`:

````markdown
### Kimi-K3 starter path

The same starter flow ships for `moonshotai/Kimi-K3`. Note: Kimi K3 weights are not yet published on HuggingFace (as of 2026-07-16); the model ID and presets are provisional mirrors of the Kimi-K2.6 starter pending the official release.

```bash
stateset-agents kimi-k3 --json-output
stateset-agents kimi-k3 --starter-profile memory --json-output
stateset-agents kimi-k3 --list-profiles --json-output
stateset-agents kimi-k3 --write-config ./kimi_k3.json
stateset-agents kimi-k3 --config ./kimi_k3.json --no-dry-run
python examples/finetune_kimi_k3_gspo.py --dry-run
```
````

5b. In the supported-models sentence, replace:

```markdown
First-class starters ship for **Qwen 3.5 0.8B**, **Gemma 4 31B IT**, **Kimi-K2.6**, **GLM 5.1**, and **GLM 5.2**.
```

with:

```markdown
First-class starters ship for **Qwen 3.5 0.8B**, **Gemma 4 31B IT**, **Kimi-K2.6**, **Kimi-K3** *(provisional)*, **GLM 5.1**, and **GLM 5.2**.
```

- [ ] **Step 6: Update `examples/README.md`**

In the `#### Kimi Models` section, replace:

```markdown
python examples/finetune_kimi_k2_6_gspo.py --dry-run
python examples/finetune_kimi_k2_6_gspo.py --starter-profile memory --dry-run
python examples/finetune_kimi_k2_6_gspo.py --list-profiles
python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task customer_service
```

with:

```markdown
python examples/finetune_kimi_k2_6_gspo.py --dry-run
python examples/finetune_kimi_k2_6_gspo.py --starter-profile memory --dry-run
python examples/finetune_kimi_k2_6_gspo.py --list-profiles
python examples/finetune_kimi_k3_gspo.py --dry-run
python examples/finetune_kimi_k3_gspo.py --list-profiles
python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task customer_service
```

And replace:

```markdown
Use `examples/finetune_kimi_k2_6_gspo.py` when you want the packaged starter path with the same `balanced`, `memory`, and `quality` preset flow as the Qwen starter.
```

with:

```markdown
Use `examples/finetune_kimi_k2_6_gspo.py` when you want the packaged starter path with the same `balanced`, `memory`, and `quality` preset flow as the Qwen starter. `examples/finetune_kimi_k3_gspo.py` is the same flow for the provisional `moonshotai/Kimi-K3` ID (HF weights pending as of 2026-07-16).
```

- [ ] **Step 7: Add the CHANGELOG entry**

In `CHANGELOG.md`, replace:

```markdown
## [Unreleased]

### Added — GLM 5.2 starter path
```

with:

```markdown
## [Unreleased]

### Added — Kimi-K3 starter path

- **`stateset_agents/training/kimi_k3_starter.py`** — packaged GSPO starter for
  `moonshotai/Kimi-K3` (provisional ID — HF weights, model card, and license not
  yet published as of 2026-07-16), mirroring the Kimi-K2.6 surface: `KimiK3Config`,
  `get_kimi_k3_config`, profile matrix (balanced/memory/quality), JSON/YAML config
  round-trip, and lazy exports from `stateset_agents.training`.
- **`stateset-agents kimi-k3`** CLI command (`cli_train.py`) and
  `stateset-agents init --preset kimi-k3` scaffold preset (`cli.py`).
- Examples: `examples/finetune_kimi_k3_gspo.py`, `examples/kimi_k3_config.py`.
- Docs: `docs/kimi_k3_starter.rst`, CLI reference section, SUPPORTED_MODELS row,
  README starter section.
- Tests: `tests/unit/test_kimi_k3_config.py`, `tests/unit/test_kimi_k3_module_exports.py`,
  `kimi-k3` command + init-preset tests in `tests/unit/test_cli.py`.

### Added — GLM 5.2 starter path
```

- [ ] **Step 8: Verify docs**

Run: `grep -nE 'K26|k2_6|[Kk]2\.6|k2-6' docs/kimi_k3_starter.rst`
Expected: no output.

Run: `grep -c "kimi_k3\|kimi-k3\|Kimi-K3" docs/index.rst docs/examples.rst docs/CLI_REFERENCE.md docs/SUPPORTED_MODELS.md README.md examples/README.md CHANGELOG.md`
Expected: every file reports a count ≥ 1.

Optional (only if sphinx is installed — `python -c "import sphinx"` succeeds): `python -m sphinx -b html docs docs/_build/html -q` and confirm no new warnings mention `kimi_k3_starter`.

- [ ] **Step 9: Commit**

```bash
git add docs/kimi_k3_starter.rst docs/index.rst docs/examples.rst docs/CLI_REFERENCE.md docs/SUPPORTED_MODELS.md README.md examples/README.md CHANGELOG.md
git commit -m "docs: add Kimi-K3 starter docs, README section, CHANGELOG entry

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Full regression and final sweep

**Files:**
- No new files. Verification only (plus fixes if anything is red).

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: a green branch ready for PR.

- [ ] **Step 1: Confirm no K2.5/K2.6 file was touched**

Run: `git diff master --name-only | grep -iE "k2_5|k25|k2_6|k2-6"`
Expected: no output — no changed file has a K2.5/K2.6 name. (Shared files like `cli.py` and `test_cli.py` gained K3 content, but their names don't match, and no `kimi_k2_6_*` / `kimi_k25_*` file may appear.)

- [ ] **Step 2: Repo-wide residual-token sweep over new files**

Run: `git diff master --name-only --diff-filter=A | xargs grep -nE 'K26|k2_6|[Kk]2\.6|k2-6'`
Expected: no output (every newly added file is free of K2.6 tokens).

- [ ] **Step 3: init preset smoke**

Run: `stateset-agents init --preset kimi-k3 --path /tmp/claude-1001/-home-dom-stateset-agents/397153ae-a8f5-4250-8ddc-508a6080a295/scratchpad/kimi_k3.json --format json && python3 -c "import json; d=json.load(open('/tmp/claude-1001/-home-dom-stateset-agents/397153ae-a8f5-4250-8ddc-508a6080a295/scratchpad/kimi_k3.json')); print(d['model_name'])"`
Expected: prints `moonshotai/Kimi-K3`

- [ ] **Step 4: Full unit suite**

Run: `python -m pytest tests/unit -q`
Expected: all pass; same skip/deselect profile as master (no new failures, no newly skipped tests).

- [ ] **Step 5: Commit any fixes (only if Steps 1-4 forced changes)**

```bash
git status --short
```

If clean: done, no commit. If fixes were needed:

```bash
git add -A
git commit -m "fix: address Kimi-K3 regression sweep findings

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
