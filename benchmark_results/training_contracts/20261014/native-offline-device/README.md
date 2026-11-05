# Native offline policy device contracts

The corrected wheel passed **8,251 tests**, with **57 optional skips**, type
checking for **383 package modules**, repository checks and both real transformer
wheel examples. `verification.json` records each stage and the unchanged source
inventory. `wheel_proof.json` identifies the rebuilt wheel by content hash.
`cpu_contract.json` passed actual logged-policy updates, exact checkpoint resume
and finite inference after reload. `cuda_contract.json` passed the same contract
on an **NVIDIA RTX A5000** with **PyTorch 2.7.1+cu128 / CUDA 12.8**: two optimizer
updates, exact agreement between resumed and uninterrupted weights, and finite
CUDA inference from the reloaded export. The downloaded policy bytes were
rehashed and their checkpoint publication revalidated locally.

The first CUDA attempt performed real policy updates, then failed while restoring
CUDA RNG states from a checkpoint loaded onto the GPU. PyTorch requires CPU byte
tensors for generator states. The shared restore path now normalizes those states,
with one- and two-generator regression tests. The first pod's deletion was
confirmed, with estimated compute of $0.0543. `attempt-1/` retains the original
CPU report, CUDA failure log, provider record and cost ledger. A transient kernel
RPC wait was observed before the process reached this error; that observation
alone does not establish a provider defect.

`attempt-2/` retains the successful run's logs, provider record and cost ledger.
`device_output.json`, `installation_output.json` and `cost_ledger.json` preserve
those records in files included by the repository's evidence rules; raw `.log`
and `.jsonl` copies are ignored by Git.
Both pods were confirmed deleted. Total estimated compute across both attempts
was **$0.0831**; the retry kept the first attempt's original one-hour deadline.
`summary.json` binds the final result to the wheel, runner and downloaded policy
hashes. Raw SFT/RL pipeline reports are retained in `wheel_pipeline/`; reusable
policy exports remain at `/tmp/stateset-native-gpu-cuda-fix-20261014/artifacts`.

This is synthetic execution evidence for an unreleased checkout based on
package version 0.66.1. It does not demonstrate representative agent quality,
production readiness, external-backend conformance, or comparative performance.
Provider compute estimates exclude storage/network charges and are not invoices.
