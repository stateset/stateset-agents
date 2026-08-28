# Provider canary evidence

These reports are non-billable, read-only authentication and cleanup probes.
They prove only the checks recorded in each JSON document, not training or
inference lifecycle success.

- River and RunPod passed on 2026-08-27.
- Fireworks was skipped because `FIREWORKS_API_KEY` and
  `FIREWORKS_ACCOUNT_ID` were not available. The skipped report is retained so
  absence cannot appear as a green provider result.

The strict publication gate currently fails by design:

```bash
python benchmarks/provider_evidence.py benchmark_results/provider_canaries
```

Use `--allow-skipped` only to validate the structure of an explicitly
incomplete matrix; its generated report has `passed: false`.
