# Fine-tuning checkpoint recovery: acceptance

1183 targeted source tests and
1072 installed-wheel tests passed with no
skips or failures (counts overlap). All 426 package modules passed typing.
Ruff, serial formatting of eleven changed Python files, repository hygiene,
release governance, stable API compatibility and workflow lint passed.

Preparation `b7168f0af46e` froze package/test Python and packaged assets,
ran source contracts and built the byte-matching wheel. Three queued wheel
groups use the same frozen root fixtures and pytest configuration and cover
exactly the selected source case identities. Final job `2c6e30457995` rechecked
frozen identities and completed real installed CLI and offline audit checks.
Sanitized queue receipts retain argv, IDs, outcomes and timings.

## Recovery behavior

Workers publish checkpoint receipts only after complete saves. Receipts bind
parent plan, training settings, dataset, progress and checkpoint inventory.
Recovery requires Linux, one local training process, matching code, a pinned
model commit, a terminal parent, no live supervisor/executor and remaining
original training budget. Model-only exports and older unbound artifacts are
unavailable. Native publications and SFT optimizer/scheduler/RNG files must be
complete. Inspections never unpickle checkpoint state or load models.

The `z` picker hashes in the background and remains bound to its captured run.
Checkpoint review authorizes only a private copy into a new linked run. A
separate prepared-run review authorizes launch. Source artifacts, stop requests
and configuration drafts are preserved. Preparation publishes no partial run on
failure; copies use separate private files, not hardlinks. Launch and execution
rehash the copied input and refuse changed training settings or dataset bytes.
Resuming preserves original total epochs/updates, with no fresh-training fallback.
Read-only inspections may be abandoned; authorized copy/launch work is owned
until it finishes. Existing prepared recovery reviews and launch validation
also run in the background.

Unit fixtures cover stale checksums, tampered receipts/weights, missing state,
symlinks/FIFOs, empty weights, incomplete native publication, changed code or
configuration, live and unverifiable processes, multi-process contexts,
exhausted budgets, byte/file limits, insufficient headroom and failed copy
cleanup. Independent snapshots survive later parent/source changes. CLI status
returns bounded recovery lineage. Strict admin HTTP inspection shares the owned
inspection slot and offers no HTTP preparation or launch route.

## Direct evidence and limits

A real tiny CPU HF/PEFT SFT run stops after its first update, publishes a
checkpoint, copies it into a recovery run and finishes the original epoch
budget. Its final adapter tensors exactly equal uninterrupted training with the
same synthetic model/data/settings. A real native CPU checkpoint restores
optimizer state, RNG and episode cursor through the copied snapshot, then
matches subsequent model/optimizer/scheduler updates. These are software
recovery contracts, not GPU performance or quality studies.

Real PTYs at 74x44 and 110x32 inspect a synthetic checkpoint, Escape the copy
review without writing a child, then explicitly prepare a child and Escape its
launch review without a worker or stop receipt. Narrow rendering is also
covered at 72x22. The installed CLI refuses a stale recovery checksum without
creating a run, prepares an exact private snapshot without launching and returns
its lineage/status. Existing guides, quality, stop, dataset, backend, worker,
reward and checkpoint-publication contracts provide regressions.

The live Herdr split was refreshed: True; its private
draft was preserved: True. It has no retained runs or
configured dataset; recovery interactions use isolated synthetic test runs.
No paid training, GPU recovery certification, remote recovery, judge execution
or independent quality gain occurred. The free recovery-study planner still
blocks paid execution. Broad historical distributed matrices were not repeated.
