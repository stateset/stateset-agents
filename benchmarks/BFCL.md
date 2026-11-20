# BFCL function calling as a training environment

`stateset_agents.training.bfcl` exposes four single-turn BFCL V4 categories
(`simple_python`, `multiple`, `parallel`, `parallel_multiple`) as an
environment whose reward is the official AST checker's verdict. No model has
been trained or scored with it yet.

## Why a separate interpreter

`bfcl-eval` pins an older numpy and a newer httpx than this package and
depends on several gigabytes of model-provider SDKs. The official decoder and
checker therefore run in their own virtual environment, and
`bfcl_checker_worker.py` talks to this package over JSON lines:

```bash
python -m venv .bfcl-venv
.bfcl-venv/bin/pip install --no-deps bfcl-eval==2026.3.23
.bfcl-venv/bin/pip install -r requirements/bfcl-checker.txt
```

```python
from stateset_agents.core.environment_registry import make_environment
from stateset_agents.training.bfcl import bfcl_split

environment = make_environment("bfcl", python=".bfcl-venv/bin/python", category="multiple")
train, test = bfcl_split(environment.task_ids(), seed=20261120)
```

The install is about 30 MB. Provider SDKs that `bfcl_eval` imports at load
time are replaced by empty stand-ins inside the worker; the checker and
decoder code is unmodified. The worker reports which packages it stood in
for as `BfclChecker.stood_in`.

## What differs from an official BFCL score

- BFCL has no training split. `bfcl_split` divides a category by a seeded
  hash so a policy is never scored on tasks it trained on. Scores on the
  held-out half are **not comparable with leaderboard numbers**.
- The system prompt is BFCL's reference prompting template filled with the
  task's functions as JSON. The official handler pipeline is not used.
- Only the four categories above are covered. Live, multi-turn, memory,
  web-search, Java and JavaScript categories are not.

## Failure handling

If the worker exits or sends a malformed reply, the environment raises
`EvaluationUnavailable`. It never scores a checker failure as a wrong answer.

The repository's tests replay 60 official reference answers per category
through the real checker when the isolated interpreter is present (set
`STATESET_BFCL_PYTHON` to its path), and skip otherwise.
