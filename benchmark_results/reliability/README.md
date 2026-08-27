# Fault-injection and recovery evidence

The A+ reliability gate requires three matching seeds for each prescribed
fault: worker exit, controller restart, and network interruption. Every run
must prove exact checkpoint replay, zero duplicate optimizer updates, bounded
lost work, completion at the expected final step, artifact integrity, and zero
remaining remote resources.

Run the validator after fault-injection jobs have retained their JSON evidence:

```bash
python benchmarks/reliability_evidence.py \
  benchmark_results/reliability/evidence \
  --max-data-loss-steps 10 \
  --output benchmark_results/reliability/report.json
```

The validator only accepts `measured: true`. It does not simulate failures.

Generate the full matrix on CUDA (or use `--device cpu` for a local smoke
test):

```bash
python benchmarks/run_reliability_matrix.py \
  --device cuda \
  --seeds 42 1337 2026 \
  --output-dir benchmark_results/reliability
```

The harness applies three distinct faults after an atomic optimizer
checkpoint: the worker exits itself, the supervising process sends SIGKILL to
the training controller, and a live TCP control-plane listener is closed
before the worker's next heartbeat. Each run launches a new process, reloads
model and optimizer state, reaches the expected final step, and verifies the
complete update ledger. Cleanup is fail-safe: child processes and listener
sockets are stopped even when the harness itself raises.

The TCP test is a single-host control-plane interruption. It does not establish
multi-node partition tolerance.
