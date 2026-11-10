# RL fan-out

Run many independent RL runs in parallel, each in its own StateSet Sandbox, from
one manifest. Results come back to a local directory and roll up into a per-arm
report.

```bash
pip install 'stateset-agents[sandbox]'
export STATESET_API_KEY=...

stateset-agents fanout plan sweep.yaml          # free: validates and prices it
stateset-agents fanout run sweep.yaml --run-dir runs/sweep                 # local
stateset-agents fanout run sweep.yaml --run-dir runs/sweep \
    --backend sandbox --confirm-max-cost-usd 3.00
```

A manifest expands to arms times seeds. Every run is isolated: its own sandbox, its
own budget, its own journal entry. A preemption or one crashing seed touches nothing
else.

```yaml
schema_version: 1
name: sandbox-sweep
seeds: [42, 43, 44]
matrix:                    # cartesian product; every combination is one arm
  updates: [40, 80]
workload:
  kind: workflow_regression
report:
  metric: improvement      # which metric ranks the arms
  mode: max
resources:
  cpus: "4"
  memory: 8Gi
  resource_class: shared   # Spot pricing, about 20x cheaper; can be preempted
  timeout_seconds: 1800
  max_cost_cents: 25       # the platform stops a sandbox at this spend
package:
  source: wheel
  wheel: dist/stateset_agents-X.Y.Z-py3-none-any.whl
  extras: [training]
concurrency: 6
max_attempts: 2
```

More in [`examples/fanout/`](../examples/fanout/).

## Workloads

| `kind` | What a run does | Metrics come from |
|---|---|---|
| `train` | `stateset_agents.training.train` in-process, then an evaluation | episode rewards and `eval_*` |
| `command` | Any program, with `{seed}`, `{output_dir}`, `{stage_dir}`, `{python}`, `{run_id}` and matrix keys substituted | `metrics.json` the program writes to `{output_dir}` |
| `workflow_regression` | Real policy updates on the offline tool-workflow task, CPU, no downloads. Validates training mechanics, not domain quality | `improvement`, success rates |

`command` is also how weight-edit sweeps run: see
[`examples/fanout/weight_edit_sweep.yaml`](../examples/fanout/weight_edit_sweep.yaml)
and [Weight editing](WEIGHT_EDITING.md#sweeping-recipes-in-parallel).

`command` reaches everything else in the repo: starters, studies, your own script
(ship it with `stage`). The `train` workload's trainer can only drive sequence-level
objectives, so it accepts `gspo` and `gspo_token`; reach the others through their
own trainers with `command`.

## Getting the code into the sandbox

| `package.source` | Start time | Use when |
|---|---|---|
| `preinstalled` | Immediate | A [custom image](https://github.com/stateset/stateset-sandbox/blob/main/docs/CUSTOM_IMAGES.md) (`resources.image: org:<name>`) already has it. Best for anything large |
| `wheel` | Installs per run | Testing a local build. Uploaded through the file API, so 7 MiB at most |
| `pypi` | Installs per run | A published release; `package.version` is required |

CPU sandboxes install CPU-only torch automatically; without that, pip fetches
gigabytes of CUDA libraries a CPU pod cannot use. The stock image's ephemeral disk
is small, so set `resources.ephemeral_storage` (20Gi works) when installing the
`training` extra. Installing needs egress to PyPI and `download.pytorch.org`, and
that download repeats for every attempt, which is another reason to prefer an image.

`resources.timeout_seconds` covers installing and running. Each sandbox is asked to
live 10 minutes longer than that, so a run that reaches its limit still has its
output collected before the sandbox goes. A plan's runtime limit can silently grant
less (Hobby's is 45 minutes). When it does, the run's limit is shortened to fit, and
the command says so.

## What it costs

`fanout run --backend sandbox` will not start until you approve a computed worst
case:

    runs x max_attempts x resources.max_cost_cents

The number comes from the per-sandbox budget the platform enforces rather than a
price table. `max_cost_cents` is required on the sandbox backend. That budget covers
compute and LLM calls. **Network transfer past your plan's allowance is billed on top
and is not in this number**, so large per-attempt downloads cost extra. Enforcement
runs periodically, so a sandbox can overshoot by up to one check interval. What each
attempt reportedly cost goes to the report and the shared cost ledger; billing
records remain authoritative.

Requests are paced across all runs to `--max-requests-per-minute`, 50 by default, to
stay under the smallest plan's limit of 60. A fan-out of many runs is polled less
often at that rate, so a finished run can sit billing a little longer before it is
noticed. On a larger plan (Pro 600, Team 6000), raise it.

## When things go wrong

| Situation | What happens |
|---|---|
| Spot preemption, lost pod, install failure, unreachable sandbox | Retries on a fresh sandbox, up to `max_attempts` |
| The workload itself fails | Final: the worker ran, and it would fail the same way again while billing again |
| Time limit or cost budget reached | Final: another attempt spends the same ceiling to reach the same end |
| Org concurrency limit or rate limit (429) | Waits for room. Uses no attempt and keeps the sandbox, so `concurrency` can safely exceed the plan's limit |
| Bad credentials, billing, a refused configuration | Stops launching, since every run would fail. Unstarted runs stay pending for `resume` |
| Ctrl-C | Stops every sandbox, then writes the report |
| The driver dies (`kill -9`, laptop closes) | Runs keep going. `fanout resume` reattaches, and collects anything that finished meanwhile; `fanout cancel` stops them |

Each attempt's create key is on disk before the create is sent, so a lost response or
a crash finds its sandbox again rather than orphaning it. The sandbox id is on disk
before anything else touches the sandbox. An ambiguous start is never repeated within
one sandbox, because the request may still have landed, so two copies can never write
the same output.

A sandbox that will not stop is reported loudly and fails the command, because it may
still be billing. So is a create whose outcome could not be learned: its cost reads
as unknown, and each later retry or `fanout resume` asks again with its key and stops
what it finds. If nothing ever settles it (a controller error can hold a key forever),
it stops being reported once the sandbox must have expired; its cost stays unknown.
Either way the sandbox's own time limit and budget bound it. A `fanout cancel` cannot
settle an unresolved create, and says so. A queued create on a plan with queue mode
holds its worker until the queue answers, and a cancel takes effect after that.

## Reading the results

```
runs/sweep/
  run.json                 what this directory is
  report.json              per-arm statistics and the verdict
  events.jsonl             everything that happened
  runs/<run_id>/
    state.json             status and every attempt
    attempt-1/stdout.log   mirrored logs; they outlive the sandbox
    attempt-1/result.json  the worker's result
    attempt-1/artifacts/   whatever `collect` matched
```

`fanout status`, `logs`, and `report` read only this directory. What each run status means, and what
can follow it, is in [Lifecycles](LIFECYCLES.md) (`stateset-agents lifecycles fanout-run`).

The verdict names a winner only when the evidence separates the best arm from every
other: all arms complete, at least two seeds each, and non-overlapping 95% intervals
(Student's t, since fan-outs usually have few seeds). Otherwise it says `unresolved`
and why. A failed or unfinished seed makes its arm incomplete rather than quietly
leaving its mean, because a seed that crashed is evidence about the arm.

`collect` patterns choose which output files come back. On a failed run they still
come back: they are often what the money bought.

**Nothing over 37 MiB can leave a sandbox through the platform's API**: file reads
are capped at 50 MB of base64, and artifact uploads go through the same read. Have
the workload push large checkpoints to your own storage, with credentials from
`forward_env`. Anything `collect` asked for that did not come back, including a
pattern that matched nothing, is named in the report and makes the command exit
nonzero, since the sandbox and the file are gone by then.

## Secrets

Manifests get committed, so `env` refuses names that look like secrets. List them
under `forward_env` instead: values are read from your environment at launch, passed
to the process, and never written to the run directory. Org secrets on the platform
also work. `STATESET_API_KEY` itself cannot be forwarded: a workload never needs the
credential that creates sandboxes. The platform refuses `AWS_*`, `GOOGLE_*`, `GCP_*`,
`AZURE_*`, `KUBERNETES_*` and `EXEC_AGENT_*`, so these are caught before anything is
allocated; pass such values under a different name.

A workload that prints a secret still puts it in the mirrored logs.

## Not included

- **Snapshot forks.** `POST /sandbox/:id/fork` takes no `budget` and only the dedicated
  class, so a forked fan-out could neither enforce per-run spend nor use Spot pricing.
  A custom image gives fast starts without giving those up.
- **Checkpoint resume across attempts.** A retried attempt starts its run over. Durable
  sandboxes could change that, at the cost of Spot's price advantage.
- **Experience sharing between runs.** Runs are independent by design. For one learner
  across many actors, see `training/distributed_rollouts.py`.

## Testing

| Test | Covers | Needs |
|---|---|---|
| `tests/unit/test_fanout_*`, `test_cli_fanout` | Manifest, report, worker, orchestration failure paths, spend gate | Nothing beyond the dev extras |
| `tests/integration/test_fanout_local_e2e.py` | Real processes and real policy updates | torch, transformers |
| `tests/integration/test_fanout_sandbox_sdk.py` | Real SDK over real HTTP against a local stand-in controller | `stateset-sandbox` |
| `tests/integration/test_fanout_sandbox_contract.py` | Requests and the stand-in match the controller's `openapi.yaml` | A sandbox repo checkout |
| `tests/integration/test_fanout_sandbox_live.py` | A real controller. Allocates real compute | `STATESET_FANOUT_LIVE=1`, an API key, a wheel |
