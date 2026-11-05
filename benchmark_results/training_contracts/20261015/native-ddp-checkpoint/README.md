# Native DDP checkpoint acceptance

Buildr job `f5f03f9d0d1d` passed the frozen source-and-wheel verifier on CPU/Gloo
with two real workers. See [summary.json](summary.json) for counts, runtime,
wheel SHA-256 and explicit evidence boundaries. [verification.json](verification.json)
pins every package and test Python source file; the wheel contains the exact
accepted package sources.

- 1,890 targeted source regressions passed, with 7 skips.
- Typing passed for all 406 source modules.
- 559 targeted installed-wheel regressions passed.
- Five real distributed pytest contracts passed against source and wheel.
- The new contract contains 84 resume and 14 rejection/recovery scenarios per
  worker, with identical source/wheel outcomes. These scenario counts are not
  additional pytest cases; source and wheel regressions overlap.

Resume covers both checkpoint APIs, SGD momentum and AdamW, the default
objective and GRPO/DR-GRPO/DAPO/CISPO/RLOO/GSPO, partial and completed windows,
and repeated inner updates. It checks exact next parameters and optimizer
state, Python/NumPy/Torch RNG draws, rank-local persistent/nonpersistent buffers,
pending policy/entropy banks and baseline history. Publication contains one
shared payload and one shard per rank.

Faults include one-rank write/read/validation/apply errors, malformed shards
inside freshly valid publication inventories, generation/topology/shared-value
mismatches, invalid local gradients/RNG/buffers, and different operations or
paths across ranks. Failed saves preserve the old generation. Validation
failures preserve live state. Apply failures block learning and saving on both
peers until a successful coordinated reload; training then proceeds.

Unit tests cover replay/private RNG state, EWC wrapper-name translation, LwF
reference restoration, missing reference relationships, optimizer slot schemas
and owned CPU optimizer storage. This is not an end-to-end certification of
DDP continual-learning task transitions or evaluation decisions.

The existing 294 update/report, 48 reward-population, 40 learner-admission and
32 completed-checkpoint startup scenarios also pass. Per-worker reports are in
[source-distributed](source-distributed/) and [wheel-distributed](wheel-distributed/).
Earlier failures and their fixes are retained in [failures/context.json](failures/context.json).
The verifier, gate helper and retention script are preserved verbatim as
`.py.txt` input artifacts alongside the logs; they are not packaged modules.

This protocol requires the same ordered process-group ranks, Python/Torch
versions, device family, supplied architecture/tokenizer/configuration and
absolute shared-filesystem path. Every peer must enter the same operation.
Dataset/environment cursors, arbitrary callback state, process/network failure
recovery and Hugging Face model export are outside this checkpoint protocol.
GPU/AMP, multi-node, multi-rank Accelerate, FSDP/DeepSpeed recovery and model
quality gains remain unverified by this record.
