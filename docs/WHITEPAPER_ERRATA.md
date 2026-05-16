# Whitepaper Errata

This file tracks corrections and clarifications to [`docs/WHITEPAPER.md`](./WHITEPAPER.md) discovered after a version was published. Each entry names the whitepaper version it corrects and the source-tree commit at which the correction was verified.

The convention: when `git log` shows commits more recent than the whitepaper's anchor commit, check this file before citing the document.

---

## Corrections to v0.11.6 (anchor commit `14c0e65`)

Resolved in the v0.12.2 revision of the whitepaper (anchor commit `c0dbd68`). Listed here for readers who hold a v0.11.6 PDF or printout.

### §3.2 — Stub backend test-count reference

> "…enables the 1,624-test suite to run in seconds without GPU hardware."

The collected test count at `c0dbd68` is **2,438** (`pytest --collect-only -q tests/`). The 1,624 figure was correct at an earlier point in the v0.11 line and was not refreshed alongside additions to `tests/unit/`, `tests/integration/`, and `tests/performance/`. The architectural claim — that the stub backend makes the suite GPU-free — is unchanged.

### §9 — Canonical exception tuples

> "(`IMPORT_EXCEPTIONS`, `GPU_EXCEPTIONS`, `MODEL_IO_EXCEPTIONS`, `INFERENCE_EXCEPTIONS`, `NETWORK_EXCEPTIONS`, `SERIALIZATION_EXCEPTIONS`, `ENVIRONMENT_EXCEPTIONS`, `SERIALIZATION_EXCEPTIONS`)"

Two errors in this list:

1. `SERIALIZATION_EXCEPTIONS` was listed twice.
2. `ENVIRONMENT_EXCEPTIONS` is not a defined name in `stateset_agents/exceptions.py`.

The correct eight tuples at `c0dbd68` are:

- `IMPORT_EXCEPTIONS`
- `GPU_EXCEPTIONS`
- `MODEL_IO_EXCEPTIONS`
- `INFERENCE_EXCEPTIONS`
- `ATTRIBUTE_VALUE_EXCEPTIONS`
- `NETWORK_EXCEPTIONS`
- `SERIALIZATION_EXCEPTIONS`
- `MODEL_DEVICE_EXCEPTIONS`

Verify with:

```bash
grep "^[A-Z_]*_EXCEPTIONS" stateset_agents/exceptions.py
```

### Appendix C.4 — Test-count comment

The comment `# Test count (claimed: 1,624)` is updated to `# Test count (claimed: 2,438)` in the v0.12.2 revision. Same underlying issue as §3.2.

---

## Corrections to v0.12.2 anchor `c0dbd68`

Re-anchored to commit `a2bdde4`. Triggered by a single-line fix in `stateset_agents/training/gspo_trainer.py` (canonical reward signature — see issue #16) that shifted lines after 596 up by one.

### §5.2 — GSPO trainer LOC count

> "GSPOTrainer (training/gspo_trainer.py, 853 LOC)"

Now **852 LOC** at `a2bdde4` (`wc -l stateset_agents/training/gspo_trainer.py`).

### §5.2 — Implementation citations

> "Clipped surrogate at gspo_trainer.py:640-650. Per-sequence KL penalty (only when beta > 0) at lines 653-661."

At `a2bdde4`:

- Clipped surrogate: **lines 639-649**
- Per-sequence KL penalty: **lines 652-660**

The sequence-ratio citation (`gspo_trainer.py:390-419`) is unchanged — those lines are upstream of the shifted region.

---

## Reporting new errata

If a `grep`, `sed`, or `wc -l` command from Appendix C disagrees with what the whitepaper says, open an issue referencing:

- The whitepaper version (front matter)
- The whitepaper's anchor commit (Versioning section)
- The current `git rev-parse HEAD`
- The specific quote and the disagreeing output

We treat the code as authoritative and update the document.
