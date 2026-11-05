# SFT split boundaries before provider setup

Declared validation and test data now fail before generic SFT training setup,
including direct in-memory SFT and industry project preparation/training. Both
top-level `split` and `metadata.split` are checked. Duplicate JSON object keys,
unknown roles and conflicting declarations fail instead of silently changing a
row's interpretation. A mixed train/held-out input fails the entire training read.
Unannotated chat and tool data remain supported. General-purpose dataset reads
can explicitly select validation or test roles.

Local SFT sources are preflighted in the shared executor contract before provider
setup, uploads or compute allocation on Local, River, RunPod, Modal, CoreWeave,
Nebius, Hugging Face Jobs and Fireworks. Tests exercise actual submit entrypoints
with guarded provider boundaries. Packaged SFT workers check their input again;
River also loads training rows before client capability checks and tokenization.
Preparing a new project may derive validation rows from declared training rows;
the new output annotations match their roles and source bytes remain unchanged.

805 unique tests passed: 52 new split-boundary tests, 596 dataset/training/provider
regressions, and 157 CLI/API/submission regressions. Full-package mypy passed for
443 source files. Ruff, Black, isort, repository hygiene and whitespace checks
passed. Black ran outside the known sandbox worker-IPC restriction.

The retained corpus review independently reconstructed all 2,112 synthetic
industry demonstrations, accepted all 1,536 training rows, completed three CPU
worker previews, and rejected 24 held-out inputs across four boundaries and three
domains. No output adapter directory, provider call or model update was created.
The dataset manifest hash references the unchanged corpus retained in the earlier
industry-studio-integration evidence. Source hashes and the selected changed
sources are retained here; earlier evidence bundles remain unchanged.

These checks enforce source declarations. They cannot identify removed or
replaced annotations or independently prove provenance. Keep source files stable
during submission: provider preflight does not bind uploaded bytes to a private
snapshot. Legacy generic SFT loading still skips malformed JSON/missing-message
rows; declaration violations and duplicate keys are fatal. Remote worker
protection requires this updated package. This review does not establish live
provider certification, production data review, model-quality gains or A+ status.
