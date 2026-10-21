# Training-only refund boundary curriculum

Validation replay identifies an inclusive-window reasoning failure in the existing six-seed River experiment: the initial SFT checkpoints failed **38 of 48 boundary cases**, despite validation using the same 14/30-day window values as training. Every one of the 1,152 recorded validation outcomes was replayed in a fresh sandbox. `validation-diagnostics.json` binds the original validation cases and histories by SHA-256 and reports all steps, including failures. This review analyzed validation traces. Earlier held-out outcomes were already exposed in prior reviews and cannot confirm the effectiveness of this change.

The new `adjacent_boundary_v1` curriculum preserves the 768 canonical training cases across six seeds and adds **747 deduplicated training-derived cases**, for **1,515 verified demonstrations**. Its 288 contrast groups connect the day before, the inclusive boundary day, and the day after the original training window. Payment, prior refunds, status, chargeback state, notes and window values remain the source training facts; only elapsed days, derived identities and family annotations change. Partial refunds continue to subtract recorded prior refunds. All original training families remain represented.

The packaged builder validates the source bundle against the canonical generator and rejects overlap with its declared validation/test cases by identity and observable-fact fingerprint. The review additionally checked all six seeds' reserved validation/test sets, all six exposed recovery primary-test sets, and all three earlier canaries' train/validation/test cases: **zero overlaps**. The 1,376 exclusion fingerprints are retained solely for identity checks. There is also no cross-seed overlap among the prepared training bundles. Held-out facts are never used as training parents or to supply teacher transcripts.

Each SFT conversation is generated from executed lookup observations and replayed through the policy sandbox. Case families and parent lineage stay outside the model messages. The separate audit reconstructs the entire dataset from canonical training parents and verifies exact cases, groups, lineage, metadata and teacher messages, as well as file hashes. Rehashing altered artifacts cannot make them pass reconstruction. Atomic publication uses the existing bundle writer and refuses a nonempty destination; source and completed-study artifacts remain unchanged.

The cached `Qwen/Qwen3.5-9B` tokenizer and the packaged River batch builder admitted all 1,515 chats without dropping or truncating a row. Full conversations occupy 518–584 tokens at a 4,096-token limit. Every one of the 4,545 assistant actions, including explicit finish, appears in the decoded supervised targets; the causal shift and token mask agree with the complete conversation. `tokenization-review.json` binds the serialized tokenizer and chat template by hash. This local batch check does not establish provider execution or learning.

The new offline commands are:

```bash
stateset-agents benchmark prepare-refund-curriculum \
  --data-dir outputs/policy-data-42 --output outputs/policy-boundary-42
stateset-agents benchmark audit-refund-curriculum \
  --data-dir outputs/policy-data-42 --output outputs/policy-boundary-42
```

The archive retains all six complete curriculum bundles, with their original artifact names. `source-and-checks.tar.gz` contains the new module/tests and updated CLI/documentation. `source-implementation.json` binds all packaged source files. `checks.json` records observed verification results, and the two JUnit files retain test outcomes. Prior study inputs are retained in the sibling `river-six-seed-recovery` evidence bundle; its `train.jsonl.txt` files restore to `train.jsonl` when reconstructing source data bundles. The runner's `outputs/` paths describe its original execution locations.

This is **training-data verification, not an improved model result or an A+ certification**. No provider calls, optimizer updates, or fresh held-out evaluations were performed. The original six-seed results and failed acceptance comparisons stand. A future experiment must freeze the changed training distribution, compare it with the original SFT baseline while accounting for extra training exposure, retain validation-only checkpoint selection, and use new excluded test cases. The original study planner/auditor does not accept this derived bundle as its canonical reference arm. Independently reviewed production data, demonstrated safety improvements, and bounded execution remain necessary before a production quality claim.
