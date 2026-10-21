# Retained training seeds: acceptance

1839 targeted source tests and 1728
installed-wheel tests passed without skips or failures. Counts overlap. All 429
package modules passed typing. Job `0901d605abfd` passed formatting, repository
lint/hygiene, release/API governance, source regressions, wheel build/install
and the first wheel group. Its second group exposed a missing curation-script
fixture in the isolated harness. The unchanged script was copied and hashed;
package/test Python stayed frozen. Job `c85dd6afa5ec` resumed the affected second
group, remaining third group, workflow lint and real CLI checks. Passed source
and first-group evidence is reused with explicit stage provenance. Frozen package
and test inventories, actual wheel bytes and selected test identities match.
Twenty-one changed Python files pass serial Black/isort checks.

The studio exposes Training seed (default 42, inclusive range 0–4294967295).
Drafts, immutable plans, launch reviews, details, comparisons and recovery retain
it. Old drafts use the default without silent rewriting; old runs lacking an
actual seed remain unknown. Invalid values produce field-addressable errors
without reading training data or echoing private values.

Local SFT seeds before model/adapter initialization and passes training/data
seeds to HF Trainer; manifests retain them. A real tiny CPU GPT2/PEFT test runs
three SFT fits from the same fixed base. Repeated seed 17 produces exactly equal
final adapter tensors; seed 29 produces different tensors. Nonzero trained LoRA
B tensors and manifests are checked. This establishes the controlled CPU
fixture, not GPU identity or pretrained-model quality.

Native RL applies Python/NumPy/Torch seeding before agent initialization and
passes the chosen value to its trainer. Controlled agent/trainer boundaries
verify startup draws and settings; existing actual native CPU checkpoint tests
remain in the broader suite. Exact recovery preserves seed settings and rejects
changes. No live large-model RL or provider training was run for this update.

The shared RemoteJobSpec seed round-trips through JSON and supported transports.
RunPod commands parse through the actual SFT argument parser; Modal/object-store
payloads and local/Hugging Face CLI paths retain it. River recording fakes verify
LoRA seed and retained pointer options. Unsupported providers reject an explicit
seed before upload/allocation; conflicting legacy/top-level River seeds fail
before creating a client. None preserves legacy provider defaults. Provider and
SDK boundaries are mocked, not live certification. Remote worker packages must
contain the feature; local source edits do not update an installed cloud image.

Report labels bind to the recorded training seed. Evaluation randomness is
retained separately in protocol settings, so repeated evaluation seeds cannot
manufacture independent training runs. The real rubric/recorder/replay pipeline
with injected judge replies tests training seed 17 and evaluation seed 99.
A mismatched bound quality report is rejected; unknown historical seeds cannot
pass the recorded study gate. Other seed artifacts remain recorded claims that
need independent provenance review.

This is the seed-feature baseline. A subsequent rendering-only correction
fixes numeric zero displayed as not set; see the
[final seed-zero acceptance](../finetune-studio-seed-zero/README.md).

Actual curses terminals at 74x44 and 110x32 load a seed-17 draft, edit to zero,
review without preparing or launching, save and exit. Narrow-pane checks and
existing evaluation/recovery workflows remain covered. Live Herdr refresh is
deferred to the final zero-value rendering acceptance. No live runs were created. Development lint, draft-harness, curses
redraw-marker and typing failures/corrections are retained separately; they
were implementation/harness failures, not queue failures. The original failed
wheel group and missing-script repair receipt are retained.

No paid run, live provider/GPU certification or independent model-quality gain
was established. Automatic checkpoint ranking/nomination, fresh test of selected
intermediates and inline judge-guided training remain separate work.
