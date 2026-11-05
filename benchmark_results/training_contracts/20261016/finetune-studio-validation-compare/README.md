# Queued checkpoint validation comparison: acceptance

729 targeted source tests and 726
matching installed-wheel tests passed without skips or failures. Counts overlap;
each suite includes all 45 comparison cases. Typing passed for 430 package
modules. Job `8d44566e0dc6` passed repository lint/hygiene, API/release governance,
serial formatting of seven changed Python files, source regressions, actual
wheel build/install, three wheel groups, workflow lint and installed CLI checks.
Package/test Python and assets stayed frozen; wheel bytes and selected case
identities match. Earlier lint/typing corrections and pre-subset-guard acceptance
are retained separately and are not counted as final acceptance.

In the evaluation view, `b` captures completed published-checkpoint validations
and queues local checkpoint hashing and judge recording replay. The detached job
survives closing the view/TUI; `g` reconnects and `r` polls retained metadata
without weight reads, replay, cancellation or resubmission. Pending requests for
the same anchor reconnect under a process-safe submission lock. Timeout or
unrecognized output stays uncertain, retaining private diagnostics and any
partial durable handle, rather than retrying blindly.

Matching cohorts require identical cases/provenance, parent/model, seed,
judge/rubric, generation and code protocols. Resource reservations may differ.
Incompatible protocols are explicitly excluded. Unverifiable snapshots,
inconsistent baseline recordings or conflicting repeated evaluations block a
recommendation. Consistent repeats merge into one checkpoint rather than add
independent evidence. Source scans fail closed above 100 entries; explicit
subsets are bounded and malformed IDs fail before queue/request creation.

Rows rank by success rate, judge mean and earliest saved step. Eligibility
requires improvement over the base with no new case-level success, policy,
truncation or required-check regression in either judge pass. A high mean cannot
hide a new failure. If no safe improvement is recorded, the base is retained.
No recommendation selects the final adapter implicitly. `c` restores the
recommended checkpoint and retained validation protocol into a separate form;
changed anchor or comparison metadata is rejected before touching form values;
review rechecks bytes, and submission still requires Enter. Training settings,
completed artifacts and quality receipts remain unchanged.

Tests use injected policy/judge replies through actual scoring, two-pass
recording and replay. They demonstrate an earlier checkpoint winning a tie,
rejection of score-high/outcome-worse candidates, safe base fallback, changed
bytes/recordings, baseline drift, conflicting repeats, private status projection,
submission races, process disappearance, concurrent callers, malformed subsets,
editor ownership and metadata-only reopening. Real curses terminals at 74x44
and 110x32 reopen a comparison, use its recommendation, review and leave without
submitting inference or changing the training draft. Layout/navigation also
passes at 72x22.

Real local Buildr job `e00e58d19b43` executed the installed-wheel verifier on
synthetic recorded evidence. Its execution receipt proves installed module
origin; the status CLI reconnects to a completed result and the expected earlier
checkpoint recommendation. The training parent is byte-identical before/after
outside the owned comparison directory. Queue submission argv/acknowledgement
and uncertain outcomes also have recording-fake tests; no remote provider,
model generation or paid training is part of this integration.

The existing idle Herdr split was refreshed to the verified code with private
draft bytes and permissions preserved; help exposes the comparison workflow.
No synthetic runs were added to the live workspace. These are validation
recommendations for captured completed cohorts, not perpetual byte validity or
quality certification. Nomination/export, fresh test of selected intermediates,
inline judge-guided training and an independent reviewed quality study remain
separate work. No GPU or model-quality gain was certified.
