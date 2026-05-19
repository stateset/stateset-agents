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

## Note on `docs/WHITEPAPER.pdf` (LaTeX-typeset, v0.13.4)

The canonical published PDF at `docs/WHITEPAPER.pdf` is **typeset with `pdfTeX` / LaTeX** (54 pages, proper table of contents, running headers, real math symbols, dotted page numbers). It was produced from `docs/WHITEPAPER.md` via a `pandoc` → LaTeX pipeline run on the v0.13.4 anchor commit.

**Minor drift between PDF and source markdown:** the LaTeX PDF was rendered slightly before the very last v0.13.4 markdown edits landed. The PDF therefore omits three small additions that the markdown source has:

- **§10.3** — the one-line audit note saying we audited the other bundled rubrics (`GSM8KReward`, `PartialCreditGSM8KReward`, `ToolCallingReward`) for similar blindness patterns.
- **§C.7** — the "Reproducing the §11.7 result" command block (Colab URL + `jupyter nbconvert --execute` invocation).
- **Front-matter PyPI callout** — the markdown uses the rewritten "✅ `pip install stateset-agents` is current as of v0.13.2" phrasing; the PDF retains an earlier wording that still correctly states the current PyPI status.

These edits will land in the next typeset cut. None of them change any headline claim; all three are small clarifications layered on top of unchanged content. Authoritative reading order is unchanged: code wins over doc; markdown source wins over PDF when they disagree on minor edits like these.

The `scripts/build_whitepaper_pdf.py` weasyprint-based build is **not** the canonical pipeline — it produces a draft preview from the current markdown source. The LaTeX-typeset PDF is what should be linked from publications, press, and external citations.

---

## Re-anchor `a2bdde4` → `4744c76`

The whitepaper acquired §11.7 ("First-Party Reproduction") and several supporting edits (§8.1 maturity matrix, front-matter, §10.3, §10.5, §11.5, §B.1) between `a2bdde4` and `4744c76`. The new anchor `4744c76` is the commit that landed §11.7 with the canonical three-seed positive-transfer result.

No line-number citations shifted in this re-anchor — `stateset_agents/training/gspo_trainer.py` is unchanged between the two commits (852 LOC; clipped surrogate lines 639-649; per-sequence KL lines 652-660 — all still valid).

The single substantive addition for readers verifying claims:

- Section §11.7 references `benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen25_05b_instruct.json` — the canonical positive result.
- The companion negative-result artifacts at `benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen35_08b.json` and `customer_support_qwen3_5_0_8b_gspo.json` are also referenced from §10.5.

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
