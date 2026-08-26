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
