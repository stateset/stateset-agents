# Fine-tuning lifecycle and native recovery

Buildr `12cf089a9943` passed 935 native/runtime regression
tests, 615 operator regression tests and 8
governance tests, plus typing for 431 package modules, formatting,
Ruff, repository hygiene, stable API/release governance and workflow lint.
No selected cases failed or skipped. This is a targeted source check, not a
complete repository or installed-wheel test run.

The changes prevent duplicate direct/supervised execution with exclusive
admission and private child coordination; preserve cooperative local stop after
supervisor loss with explicit uncertainty; and preflight native local learner
state, retaining quarantine after late restore/publication failures. Regression
tests include actual tiny CPU single/multi-turn continuation, pending/completed
gradient windows and AMP state, malformed state and injected late failures,
mixed admission races, and TUI/CLI/in-memory HTTP stop behavior. Provider work
uses injected stubs; this run neither provisions nor validates remote training.

Package/test hashes were frozen after formatting and held unchanged throughout
verification. The refreshed live Herdr pane matches those hashes. Saved draft
bytes, permissions and retained run history were preserved, and no training was
launched during the refresh. Staged manifests describe historical preformatter
inputs; current verified bytes are retained separately.

Model-only imports can change weights and return False; they cannot establish
learner recovery. Late failures are contained, not rolled back. Legacy complete
checkpoints without RNG permit best-effort recovery with an explicit warning.
Configured rollout synchronization opt-out remains supported. The collective
native DDP protocol is separate; these source checks do not establish new GPU,
distributed, provider or model-quality evidence.

Broader source/wheel/CLI acceptance remains separate under `1e204ba66630`
(state at retention: queued); this bundle does not adopt its outcome.
