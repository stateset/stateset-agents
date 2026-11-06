# Learning-gate calibration, 2026-11-06

The released v0.71.7 learning fixtures had four reproducible failures: online
DPO's seed-0 negative control and three seed-1 on-policy distillation checks.
The same failures reproduced after the context-budget fix in `d121ae50`.
These fixtures use short prompts and never hit that budget.

This experiment changes the toy workloads' optimization settings. It does
not change trainer implementations or defaults, remove seeds, skip controls,
or lower the held-out thresholds.

## Selection and evidence

- [calibration.json](calibration.json) records the plans and training-prompt
  measurements, including unsuccessful trials.
- [heldout.json](heldout.json) records the subsequent held-out measurements.
- Models, tokenizers, reward, data generation and held-out prompts come from
  the existing `tests/learning` fixtures. Calibration uses only training
  prompts; it does not call the fixtures' held-out evaluation helpers.
- Calibration runs use CPU PyTorch 2.11.0, Transformers 5.14.1 and Python
  3.12.12 with one Torch thread. The held-out evidence records its environment.

For online DPO, use the 12 existing training prompts, eight samples per
prompt, four epochs per round, a learning rate of 3e-3, and either three or
six rounds. After each round, measure the training digit rate with 32 samples
per prompt and the trainer's fixed evaluation seed. Six rounds satisfy the
prespecified training criteria on seeds 0/1/2: positive rates at least 0.8
under both reference schedules and negative-control rates at most 0.02.
Three rounds do not.

For distillation, use the existing teacher and its four samples per training
prompt. Measure teacher-forced forward KL and student-sampled reverse KL on
those training prompts, at temperature 1, with the live teacher. Use the
same token weighting as `DistillationTrainer.evaluate`, evaluating under
`no_grad` and preserving the training RNG and model mode. Select a budget
only when both training KLs are at most 35% of their initial values across
all three seeds.

The record includes the original rates at 30/60/120 updates, smaller updates
at 30/60/120/240, and sampled-estimator investigations: a larger batch, a
detached per-state KL baseline prototype, temperature 2, and a longer run.
The baseline prototype was isolated to the experiment; it is not a package
change. Full-distribution JSD and reverse KL converge at 1e-3 and 240 updates.
The sampled estimator's seed 1 remains stalled at 480 updates and converges
by 960 updates at 1e-3. Both other seeds also satisfy the training criteria
at that fixed budget before held-out evaluation.

The retained held-out requirements are:

- Online DPO: digit rate increases by at least 0.1 with the positive judge
  and decreases by at least 0.05 with the negative judge.
- Distillation: both KLs decrease by at least 65%, with first-token digit
  mass above 0.6.
- Shuffled-teacher controls: neither KL to the real teacher decreases, and
  digit mass stays below 0.4. Controls use the same budget as their positive
  counterparts.

## Reproduce the final gates

From the repository root with the development dependencies installed:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LEARNING_SEEDS=0,1,2 \
  .venv/bin/pytest -q tests/learning/test_online_dpo_learns.py \
  tests/learning/test_distillation_learns.py
```

These are small CPU convergence checks. They do not establish large-model
quality, and the longer on-policy workloads increase test runtime.

## Held-out verification

All 36 cases pass on PyTorch 2.11.0 and 2.13.0: nine online-DPO cases, 21
positive distillation cases, and six shuffled-teacher controls. The 2.11.0
pytest run took 89 seconds with four workers; timings depend on machine load.
The 2.13.0 measurements in `heldout.json` were collected by invoking the same
fixture helpers and checking all the original quality assertions.

The repository-wide retry recorded 16,988 passed tests and 137 skips, plus
one teardown error caused by a UI test's process-wide `os.scandir` mock.
Scoping the mock to the browsing steps fixes the error; all 12 tests in that
file pass with immediate temporary-directory cleanup enabled. The full
suite was not repeated after that isolated teardown fix.

The earlier coverage run exhausted filesystem space and ended with a
pytest-cov SQLite error. It produced no complete coverage report and is not
counted as a passing run. Task-generated temporary files were removed, and
the retry ran without coverage and with passed-test directories cleaned
immediately. [verification.json](verification.json) preserves these outcomes
separately; no new coverage percentage is claimed.
