# Getting started with StateSet Agents on StateSet Swarm

Run one reinforcement learning experiment and one weight edit in separate CPU
sandboxes. Each task returns JSON to its own directory on the swarm's results
branch. You can inspect a recorded run or execute the same task scripts locally.

| Recipe | One sandbox per | Evidence |
| --- | --- | --- |
| `starter` | Seed 42 of a GRPO refund workflow, and one checkpoint edit | Success on 128 paired synthetic cases before and after training, optimizer steps, and checkpoint reload. The edit scales one token row in a tiny Qwen2 checkpoint and checks every tensor against the edit specification. |
| `rl-seeds` | Seed 42, 43, or 44 | The same 20-update training budget for each seed. Each worker returns the success rates and individual case outcomes. |
| `qwen-layers` | Layer 6 or 14 of Qwen3-0.6B | Both workers try the counterfactual France → Lyon edit on the same model revision and recipe. Each returns generated answers, preservation metrics and the recipe decision. An accepted edit also goes through adapter reload verification. |

## 1. Inspect a result and plan a run

Start in the root of this checkout with Python 3.10+ and Git. The host CLI
renders the task scripts into the plan. The workers run the committed
`v0.73.2` source, which contains the GRPO worker and weight-edit commands.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
SWARM_CTL="$PWD/.venv/bin/stateset-agents"
WORK=$(mktemp -d)

"$SWARM_CTL" swarm list
"$SWARM_CTL" swarm report starter --results examples/swarm/evidence/local-cpu
"$SWARM_CTL" swarm plan starter > "$WORK/starter.json"
cat "$WORK/starter.json"
```

Planning needs no credentials. If the tag is missing from a shallow clone,
fetch it with `git fetch origin tag v0.73.2`. To test another committed
worker revision, pass `--base <ref>`. The plan records its resolved commit SHA.

## 2. Replay the starter locally

The first two commands export the same Bash scripts the coordinator gives to
its agents. Both workloads run on CPU with locally constructed models.
The following setup installs CPU PyTorch on Linux.

```bash
"$SWARM_CTL" swarm script rl-seed-42 > "$WORK/rl.sh"
"$SWARM_CTL" swarm script checkpoint-edit > "$WORK/edit.sh"
git worktree add --detach "$WORK/worker" v0.73.2
(
  set -e
  cd "$WORK/worker"
  python3 -m venv "$WORK/worker-venv"
  source "$WORK/worker-venv/bin/activate"
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python -m pip install '.[training]'
  bash "$WORK/rl.sh"
  bash "$WORK/edit.sh"
)
"$SWARM_CTL" swarm report starter --results "$WORK/worker/swarm-results"
```

## 3. Set up the hosted coordinator

The native coordinator runs on your machine and launches a coding agent in
each sandbox. Use Node.js 22.12 or later, a funded StateSet Sandbox
account, a CLI profile or `STATESET_API_KEY`, and model access configured
for the workers. This plan sets a $1 model budget per task. Sandbox compute
is billed separately. The Qwen sweep also downloads the pinned model in each
sandbox. [Coordinator setup and model access](https://github.com/stateset/stateset-sandbox/blob/v2.14.1/examples/claude-agent/README.md#model-access)

```bash
git clone --branch v2.14.1 --depth 1 https://github.com/stateset/stateset-sandbox.git "$WORK/coordinator"
(
  set -e
  cd "$WORK/coordinator"
  npm ci
  npm run build -w sdk
  npm run build -w cli
)
COORD="$WORK/coordinator/examples/claude-agent/swarm.mjs"
```

Choose the starter below or go directly to [the three-seed RL sweep](#4-fan-out-rl-across-three-seeds).

### Launch the starter

```bash
RUN="$WORK/starter-$(date +%Y%m%d-%H%M%S)"
node "$COORD" run "$WORK/starter.json" "$RUN" --max-parallel 2 --publish
```

The coordinator merges returned files onto a local Git branch. `--publish`
also records task state in the organization's swarm service. To watch from a
second terminal, use `node "$COORD" status "$RUN"` there with the same paths.
To resume an interrupted run, repeat the launch command with the same
plan and run directory.

```bash
node "$COORD" status "$RUN"
git worktree add --detach "$RUN-results" "swarm/${RUN##*/}"
"$SWARM_CTL" swarm report starter --results "$RUN-results/swarm-results"
```

`swarm report` exits nonzero when an expected task has missing or unreadable
evidence. For RL, inspect `optimizer_updates`, `reload_verified`, and the
signed `improvement`. For Qwen, inspect the recipe's `status` and the
individual violations in `result.json`. A worker can successfully complete
a trial whose proposed edit the recipe rejects.

## 4. Fan out RL across three seeds

From the host checkout with `SWARM_CTL` and `COORD` set above, generate a
plan and inspect its tasks before launching them. Each seed gets its own CPU
sandbox, synthetic refund environment, and 20 GRPO updates. Three workers can
run concurrently. The hosted sweep allows $1 of model spend per worker plus
sandbox compute.

```bash
RUN=$(mktemp -d "${TMPDIR:-/tmp}/stateset-rl-XXXXXXXX")
"$SWARM_CTL" swarm plan rl-seeds > "$RUN.plan.json"
python -c 'import json,sys; print([t["id"] for t in json.load(open(sys.argv[1]))["tasks"]])' "$RUN.plan.json"
node "$COORD" run "$RUN.plan.json" "$RUN" --max-parallel 3 --publish
node "$COORD" status "$RUN"
git worktree add --detach "$RUN-results" "swarm/${RUN##*/}"
"$SWARM_CTL" swarm report rl-seeds --results "$RUN-results/swarm-results"
```

The plan contains `rl-seed-42`, `rl-seed-43`, and `rl-seed-44`. Compare
`baseline_success` and `candidate_success` for each seed, then check
`optimizer_updates` and `reload_verified`. Each worker's `workflow.json`
holds the individual outcomes. To inspect the recorded local sweep immediately,
run `stateset-agents swarm report rl-seeds --results examples/swarm/evidence/local-cpu`.

## What the runs showed

On the local CPU run, seed 42 improved from 67 to 86 completed workflows
out of 128. Seed 43 improved from 63 to 89 and seed 44 fell from 64 to 41.
All three reloaded their trained checkpoints. The starter's separate edit
halved the eight weights in token row 1, from `[1, 2, …, 8]` to
`[0.5, 1, …, 4]`. [Local CPU receipts](evidence/local-cpu/)

The local Qwen layer sweep retained both previously correct locality answers
at each site. Each edit answered two of four target prompts correctly and
exceeded the permitted drift. Both were rejected. [Layer 6](evidence/local-cpu/qwen-layer-6/result.json) · [Layer 14](evidence/local-cpu/qwen-layer-14/result.json)

A prior StateSet Sandbox run on November 12 used a larger, 30-sentence
calibration recipe. The edited Qwen3-0.6B answered “Lyon” on all ten target
prompts and retained all three locality answers. That run checked the edit
in memory and skipped adapter reload. The report and original recipe are
preserved from commit `771f45a4`. [Recorded sandbox trial](evidence/france-lyon-sandbox.json) · [Pilot recipe](france-lyon-cal30.json)

To replay that pilot after step 2, run this from the host checkout. The
revision override selects the same Hub snapshot used in the recorded trial.

```bash
PILOT="$PWD/examples/swarm/france-lyon-cal30.json"
(
  cd "$WORK/worker" || exit 1
  "$WORK/worker-venv/bin/python" -m stateset_agents.cli weight-edit teach "$PILOT" \
    --set revision=c1899de289a04d12100db370d81485cdf75e47ca \
    --threads 2 --no-verify --out "$WORK/lyon-edit" --metrics "$WORK/lyon-metrics.json"
)
```

## Add an experiment

For another seed or layer, extend the corresponding tuple in the catalog.
For a different workload, add a Bash body beside the existing ones. It runs
from the worker checkout with `$scratch` for temporary files and `$out` for
returned evidence. Write `$out/result.json` and exit nonzero on execution
failure. Register the task and its worker source requirement in a recipe,
then add its summary fields to `swarm report`. Run `swarm script` locally
and `swarm plan` to inspect the coordinator input.
[Catalog](../../stateset_agents/cli_swarm.py) · [Task scripts](../../stateset_agents/swarm/)
